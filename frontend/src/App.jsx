import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { auth, db } from "./firebase";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signOut,
} from "firebase/auth";
import {
  collection,
  addDoc,
  getDocs,
  query,
  orderBy,
  doc,
  updateDoc,
  serverTimestamp,
} from "firebase/firestore";
import "./App.css";
import reactLogo from "./assets/logo.svg";

function App() {
  const [user, setUser] = useState(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [currentSessionTitle, setCurrentSessionTitle] = useState("");
  const [loadingSessions, setLoadingSessions] = useState(false);
  const chatBoxRef = useRef(null); // ✅ 스크롤용 ref

  const API_BASE = "/api";

  // 🔹 로그인 상태 감지
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      if (currentUser) {
        loadSessions(currentUser.uid);
        setMessages([]);
        setCurrentSessionId(null);
        setCurrentSessionTitle("");
      } else {
        setSessions([]);
        setMessages([]);
        setCurrentSessionId(null);
        setCurrentSessionTitle("");
      }
    });
    return () => unsubscribe();
  }, []);

  // 🔹 Firestore에서 세션 및 대화 내역 불러오기
  const loadSessions = async (uid) => {
    setLoadingSessions(true);
    try {
      const sessionQuery = query(
        collection(db, "users", uid, "sessions"),
        orderBy("updatedAt", "desc")
      );
      const snapshot = await getDocs(sessionQuery);
      const sessionList = snapshot.docs.map((sessionDoc) => {
        const data = sessionDoc.data();
        return {
          id: sessionDoc.id,
          title: data.title || "이전 채팅",
        };
      });
      setSessions(sessionList);
    } catch (error) {
      console.error("세션 목록을 불러오는 중 오류:", error);
    } finally {
      setLoadingSessions(false);
    }
  };

  const loadSessionMessages = async (uid, sessionId) => {
    try {
      const messagesQuery = query(
        collection(db, "users", uid, "sessions", sessionId, "messages"),
        orderBy("timestamp", "asc")
      );
      const snapshot = await getDocs(messagesQuery);
      const history = snapshot.docs.map((messageDoc) => {
        const data = messageDoc.data();
        return {
          role: data.role,
          text: data.text,
        };
      });
      setMessages(history);
    } catch (error) {
      console.error("세션 대화를 불러오는 중 오류:", error);
    }
  };

  const handleSelectSession = async (sessionId) => {
    if (!user) return;
    setCurrentSessionId(sessionId);
    const selected = sessions.find((session) => session.id === sessionId);
    setCurrentSessionTitle(selected?.title || "");
    setMessages([]);
    await loadSessionMessages(user.uid, sessionId);
  };

  const handleNewSession = () => {
    setCurrentSessionId(null);
    setCurrentSessionTitle("");
    setMessages([]);
  };

  // 🔹 메시지가 바뀔 때마다 자동 스크롤
  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTo({
        top: chatBoxRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages, loading]);

  // 🔹 로그인 or 회원가입
  const handleLogin = async () => {
    try {
      await signInWithEmailAndPassword(auth, email, password);
    } catch {
      await createUserWithEmailAndPassword(auth, email, password);
    }
  };

  // 🔹 로그아웃
  const handleLogout = async () => {
    await signOut(auth);
    setUser(null);
    setMessages([]);
    setSessions([]);
    setCurrentSessionId(null);
    setCurrentSessionTitle("");
  };

  // 🔹 메시지 전송
  const handleSend = async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput) return;
    setLoading(true);

    const userMsg = { role: "user", text: trimmedInput };
    setMessages((prev) => [...prev, userMsg]);

    setInput("");

    const initialTitle =
      trimmedInput.length > 20 ? `${trimmedInput.slice(0, 20)}...` : trimmedInput;

    let sessionId = currentSessionId;
    let sessionTitleForList = currentSessionTitle || initialTitle;
    let isNewSession = false;

    if (user && !sessionId) {
      const newSessionTitle = initialTitle || "새 채팅";
      try {
        const sessionRef = await addDoc(
          collection(db, "users", user.uid, "sessions"),
          {
            title: newSessionTitle,
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp(),
          }
        );
        sessionId = sessionRef.id;
        isNewSession = true;
        sessionTitleForList = newSessionTitle;
        setCurrentSessionId(sessionId);
        setCurrentSessionTitle(newSessionTitle);
        setSessions((prev) => [
          { id: sessionId, title: newSessionTitle },
          ...prev.filter((session) => session.id !== sessionId),
        ]);
      } catch (error) {
        console.error("새 세션 생성 중 오류:", error);
      }
    }

    if (user && sessionId) {
      try {
        await addDoc(
          collection(db, "users", user.uid, "sessions", sessionId, "messages"),
          {
            role: "user",
            text: trimmedInput,
            timestamp: serverTimestamp(),
          }
        );
      } catch (error) {
        console.error("사용자 메시지 저장 오류:", error);
      }
    }

    // 3초 딜레이 (타이핑 효과)
    await new Promise((resolve) => setTimeout(resolve, 3000));

    try {
      const res = await axios.post(`${API_BASE}/chat`, { message: trimmedInput });
      const answer = res.data.answer;
      const botMsg = { role: "bot", text: answer };
      setMessages((prev) => [...prev, botMsg]);

      if (user) {
        if (sessionId) {
          try {
            await addDoc(
              collection(db, "users", user.uid, "sessions", sessionId, "messages"),
              {
                role: "bot",
                text: answer,
                timestamp: serverTimestamp(),
              }
            );
            const sessionRef = doc(db, "users", user.uid, "sessions", sessionId);
            const updatePayload = { updatedAt: serverTimestamp() };
            if (isNewSession) {
              updatePayload.title = sessionTitleForList;
            }
            await updateDoc(sessionRef, updatePayload);
            setSessions((prev) => {
              const existing = prev.find((session) => session.id === sessionId);
              const filtered = prev.filter((session) => session.id !== sessionId);
              const nextTitle = isNewSession
                ? sessionTitleForList
                : existing?.title || currentSessionTitle || sessionTitleForList;
              return [{ id: sessionId, title: nextTitle }, ...filtered];
            });
          } catch (error) {
            console.error("봇 메시지 저장 오류:", error);
          }
        }
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "⚠️ 서버와의 연결에 문제가 발생했습니다." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // 🔹 로그인 화면
  if (!user) {
    return (
      <div className="login-container">
        <div className="login-box fade-in">
          <div className="login-logo">
            <img src={reactLogo} alt="React Logo" className="react-logo" />
            <h1>NA Chat</h1>
          </div>
          <p className="login-subtitle">대한민국 정책 관련 질의응답을 시작하세요.</p>

          <input
            className="input"
            placeholder="이메일"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <input
            className="input"
            type="password"
            placeholder="비밀번호"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <button className="button" onClick={handleLogin}>
            로그인 / 회원가입
          </button>
        </div>
      </div>
    );
  }

  // 🔹 메인 채팅 화면
  return (
    <div className="chat-page">
      <aside className="sidebar">
        <button
          type="button"
          className="new-session-button"
          onClick={handleNewSession}
        >
          + 새 채팅
        </button>
        <div className="session-list">
          {loadingSessions ? (
            <div className="session-placeholder">불러오는 중...</div>
          ) : sessions.length === 0 ? (
            <div className="session-placeholder">이전 채팅이 없습니다.</div>
          ) : (
            sessions.map((session) => (
              <button
                type="button"
                key={session.id}
                className={`session-item ${
                  currentSessionId === session.id ? "active" : ""
                }`}
                onClick={() => handleSelectSession(session.id)}
              >
                {session.title}
              </button>
            ))
          )}
        </div>
      </aside>

      <div className="chat-container">
        <div className="header">
          <span>National Assembly Q&A</span>
          <button className="logout" onClick={handleLogout}>
            로그아웃
          </button>
        </div>

        <div className="chat-box" ref={chatBoxRef}>
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>
              {m.text}
            </div>
          ))}

          {/* 🔹 로딩 중일 때 점 1~3개 순환 애니메이션 */}
          {loading && <div className="message bot loading-bubble"></div>}
        </div>

        <div className="input-area">
          <input
            className="input-box"
            placeholder="질문을 입력하세요..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <button
            className="send-button"
            onClick={handleSend}
            disabled={loading}
          >
            전송
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
