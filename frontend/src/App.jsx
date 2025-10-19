import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { auth } from "./firebase";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signOut,
} from "firebase/auth";
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
  const [isTemplatePanelOpen, setIsTemplatePanelOpen] = useState(false);
  const [selectedTemplateId, setSelectedTemplateId] = useState(null);
  const [templateValues, setTemplateValues] = useState({});
  const [templateError, setTemplateError] = useState("");
  const chatBoxRef = useRef(null); 

  const examplePrompts = [
    "최근 국회에서 통과된 주요 법안은 무엇인가요?",
    "청년 창업 지원 정책에 대해 알려주세요.",
    "국회 회의록을 어디서 열람할 수 있나요?",
    "지역구 의원에게 민원을 전달하려면 어떻게 하나요?",
  ];

  const templates = [
    {
      id: "bill-status",
      title: "법안 진행 상황 문의",
      description: "특정 법안의 심의 단계와 향후 일정을 확인합니다.",
      pattern:
        "{billName} 법안의 현재 국회 심의 상황과 앞으로 예정된 일정이 어떻게 되나요?",
      fields: [
        {
          name: "billName",
          label: "법안 명칭",
          placeholder: "예: 전세사기 방지법",
        },
      ],
    },
    {
      id: "policy-support",
      title: "지원 정책 상세 요청",
      description: "대상과 목적을 지정해 지원 정책 내용을 안내받습니다.",
      pattern:
        "{targetGroup}을(를) 위한 {policyTopic} 관련 정부 및 국회 지원 정책과 신청 절차를 알려주세요.",
      fields: [
        {
          name: "targetGroup",
          label: "대상",
          placeholder: "예: 청년 창업가",
        },
        {
          name: "policyTopic",
          label: "정책 주제",
          placeholder: "예: 초기 자금 지원",
        },
      ],
    },
    {
      id: "regional-issue",
      title: "지역 민원 전달",
      description: "관심 지역의 현안을 의원에게 어떻게 전달할지 묻습니다.",
      pattern:
        "{region} 지역의 {issueDetail} 문제를 담당 상임위나 지역구 의원에게 전달하려면 어떤 절차를 따르면 될까요?",
      fields: [
        {
          name: "region",
          label: "지역",
          placeholder: "예: 서울 강서구",
        },
        {
          name: "issueDetail",
          label: "현안 내용",
          placeholder: "예: 노후 주거 환경 개선",
        },
      ],
    },
  ];

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

  // 🔹 API로 세션 목록 불러오기
  const loadSessions = async (uid) => {
    setLoadingSessions(true);
    try {
      const res = await axios.get(`${API_BASE}/users/${uid}/sessions`);
      setSessions(res.data.sessions || []);
    } catch (error) {
      console.error("세션 목록을 불러오는 중 오류:", error);
    } finally {
      setLoadingSessions(false);
    }
  };

  const loadSessionMessages = async (uid, sessionId) => {
    try {
      const res = await axios.get(
        `${API_BASE}/users/${uid}/sessions/${sessionId}/messages`
      );
      const history = (res.data.messages || []).map((msg) => ({
        role: msg.role,
        text: msg.text,
      }));
      setMessages(history);
    } catch (error) {
      console.error("세션 대화를 불러오는 중 오류:", error);
    }
  };

  const selectedTemplate = selectedTemplateId
    ? templates.find((template) => template.id === selectedTemplateId)
    : null;

  const toggleTemplatePanel = () => {
    setIsTemplatePanelOpen((prev) => {
      const next = !prev;
      if (!next) {
        setSelectedTemplateId(null);
        setTemplateValues({});
        setTemplateError("");
      }
      return next;
    });
  };

  const handleSelectTemplate = (templateId) => {
    const template = templates.find((item) => item.id === templateId);
    setSelectedTemplateId(templateId);
    if (template) {
      const initialValues = {};
      template.fields.forEach((field) => {
        initialValues[field.name] = "";
      });
      setTemplateValues(initialValues);
    } else {
      setTemplateValues({});
    }
    setTemplateError("");
  };

  const handleTemplateFieldChange = (name, value) => {
    setTemplateValues((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const buildTemplatePrompt = (template, values) => {
    let text = template.pattern;
    template.fields.forEach((field) => {
      const replacement = (values[field.name] || "").trim();
      text = text.replaceAll(`{${field.name}}`, replacement);
    });
    return text.replace(/\s+/g, " ").trim();
  };

  const validateTemplateValues = (template, values) =>
    template.fields.every((field) => (values[field.name] || "").trim());

  const handleTemplateApply = () => {
    if (!selectedTemplate) return;
    if (!validateTemplateValues(selectedTemplate, templateValues)) {
      setTemplateError("모든 항목을 입력해주세요.");
      return;
    }
    const prompt = buildTemplatePrompt(selectedTemplate, templateValues);
    setInput(prompt);
    setTemplateError("");
    setIsTemplatePanelOpen(false);
    setSelectedTemplateId(null);
    setTemplateValues({});
  };

  const handleTemplateSend = async () => {
    if (loading) return;
    if (!selectedTemplate) return;
    if (!validateTemplateValues(selectedTemplate, templateValues)) {
      setTemplateError("모든 항목을 입력해주세요.");
      return;
    }
    const prompt = buildTemplatePrompt(selectedTemplate, templateValues);
    setTemplateError("");
    setInput(prompt);
    setIsTemplatePanelOpen(false);
    setSelectedTemplateId(null);
    setTemplateValues({});
    await handleSend(prompt);
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

  const handleExampleClick = async (prompt) => {
    if (!user) {
      setInput(prompt);
      return;
    }
    if (!loading) {
      setInput(prompt);
      await handleSend(prompt);
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
  const handleSend = async (overrideInput) => {
    if (loading) return;
    const rawInput = typeof overrideInput === "string" ? overrideInput : input;
    const trimmedInput = rawInput.trim();
    if (!trimmedInput) return;
    if (!user) {
      console.error("로그인 후 이용 가능합니다.");
      setInput(trimmedInput);
      return;
    }
    setLoading(true);
    const userMsg = { role: "user", text: trimmedInput };
    setMessages((prev) => [...prev, userMsg]);

    setInput("");

    const initialTitle =
      trimmedInput.length > 20 ? `${trimmedInput.slice(0, 20)}...` : trimmedInput;

    let sessionId = currentSessionId;
    let sessionTitleForList = currentSessionTitle || initialTitle;

    if (sessionId && !sessionTitleForList) {
      const existing = sessions.find((session) => session.id === sessionId);
      sessionTitleForList = existing?.title || "이전 채팅";
    }

    // 3초 딜레이 (타이핑 효과)
    await new Promise((resolve) => setTimeout(resolve, 3000));

    try {
      const payload = {
        uid: user.uid,
        message: trimmedInput,
        sessionId: sessionId || undefined,
        sessionTitle: sessionTitleForList || undefined,
      };
      const res = await axios.post(`${API_BASE}/chat`, payload);
      const answer = res.data.answer;
      const newSessionId = res.data.sessionId || sessionId;
      const newSessionTitle = res.data.title || sessionTitleForList;

      const botMsg = { role: "bot", text: answer };
      setMessages((prev) => [...prev, botMsg]);

      if (newSessionId && newSessionId !== currentSessionId) {
        setCurrentSessionId(newSessionId);
      }
      if (newSessionTitle && newSessionTitle !== currentSessionTitle) {
        setCurrentSessionTitle(newSessionTitle);
      }
      await loadSessions(user.uid);
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

        <div className="prompt-examples">
          {examplePrompts.map((prompt, index) => (
            <button
              type="button"
              key={`${index}-${prompt}`}
              className="prompt-chip"
              onClick={() => handleExampleClick(prompt)}
              disabled={loading}
            >
              {prompt}
            </button>
          ))}
        </div>

        <div className="template-toggle">
          <button
            type="button"
            className="template-toggle-button"
            onClick={toggleTemplatePanel}
          >
            {isTemplatePanelOpen ? "질의 템플릿 닫기" : "질의 템플릿 열기"}
          </button>
          {selectedTemplate && isTemplatePanelOpen && (
            <span className="template-selected-label">
              선택한 템플릿: {selectedTemplate.title}
            </span>
          )}
        </div>

        {isTemplatePanelOpen && (
          <div className="template-panel">
            <div className="template-list">
              {templates.map((template) => (
                <button
                  type="button"
                  key={template.id}
                  className={`template-card ${
                    selectedTemplateId === template.id ? "active" : ""
                  }`}
                  onClick={() => handleSelectTemplate(template.id)}
                >
                  <h4>{template.title}</h4>
                  <p>{template.description}</p>
                </button>
              ))}
            </div>

            {selectedTemplate && (
              <div className="template-form">
                <div className="template-form-fields">
                  {selectedTemplate.fields.map((field) => (
                    <label key={field.name} className="template-field">
                      <span>{field.label}</span>
                      <input
                        type="text"
                        placeholder={field.placeholder}
                        value={templateValues[field.name] || ""}
                        onChange={(e) =>
                          handleTemplateFieldChange(field.name, e.target.value)
                        }
                      />
                    </label>
                  ))}
                </div>
                {templateError && (
                  <div className="template-error">{templateError}</div>
                )}
                <div className="template-actions">
                  <button
                    type="button"
                    className="template-action secondary"
                    onClick={handleTemplateApply}
                  >
                    입력창에 적용
                  </button>
                  <button
                    type="button"
                    className="template-action primary"
                    onClick={handleTemplateSend}
                    disabled={loading}
                  >
                    바로 전송
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

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
