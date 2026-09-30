import {
  AlertTriangle,
  ArrowRight,
  HardHat,
  ShieldCheck,
  Zap,
} from "lucide-react";

interface WelcomePanelProps {
  onQuestionSelect: (question: string) => void;
}

const suggestedQuestions = [
  {
    title: "Confined Space Safety",
    description:
      "Identify hazards, risks and required safety precautions.",
    question:
      "What are the main dangers of working inside a confined space?",
    icon: AlertTriangle,
  },
  {
    title: "Personal Protective Equipment",
    description:
      "Review appropriate PPE requirements for workplace hazards.",
    question:
      "What personal protective equipment should workers use?",
    icon: HardHat,
  },
  {
    title: "Working at Height",
    description:
      "Understand precautions for safe work at elevated locations.",
    question:
      "What precautions should be taken when working at height?",
    icon: ShieldCheck,
  },
  {
    title: "Electrical Safety",
    description:
      "Explore workplace electrical hazards and safety controls.",
    question:
      "What precautions should workers take when working with electricity?",
    icon: Zap,
  },
];

function WelcomePanel({
  onQuestionSelect,
}: WelcomePanelProps) {
  return (
    <section className="welcome-panel">
      <div className="welcome-badge">
        <ShieldCheck size={15} />
        INDUSTRIAL HSE KNOWLEDGE SYSTEM
      </div>

      <h2>
        Safety intelligence grounded in
        verified HSE documents.
      </h2>

      <p className="welcome-description">
        Ask workplace health and safety questions and
        receive evidence-grounded answers with traceable
        document sources and page references.
      </p>

      <div className="suggested-section">
        <div className="suggested-heading">
          <span>QUICK SAFETY QUERIES</span>
          <p>
            Select a topic or enter your own HSE question below.
          </p>
        </div>

        <div className="suggestion-grid">
          {suggestedQuestions.map((item) => {
            const Icon = item.icon;

            return (
              <button
                type="button"
                className="suggestion-card"
                key={item.title}
                onClick={() =>
                  onQuestionSelect(item.question)
                }
              >
                <div className="suggestion-icon">
                  <Icon size={21} />
                </div>

                <div className="suggestion-content">
                  <strong>{item.title}</strong>
                  <p>{item.description}</p>
                </div>

                <ArrowRight
                  className="suggestion-arrow"
                  size={18}
                />
              </button>
            );
          })}
        </div>
      </div>

      <div className="welcome-trust-bar">
        <div>
          <ShieldCheck size={16} />
          <span>Evidence Grounded</span>
        </div>

        <div>
          <span className="trust-dot" />
          <span>Source Traceability</span>
        </div>

        <div>
          <span className="trust-dot" />
          <span>Persistent Sessions</span>
        </div>
      </div>
    </section>
  );
}

export default WelcomePanel;