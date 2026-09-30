import {
  Activity,
  Database,
  FileText,
  ShieldCheck,
  Wifi,
  WifiOff,
} from "lucide-react";

interface HeaderProps {
  isOnline: boolean;
  documents?: number;
  chunks?: number;
}

function Header({
  isOnline,
  documents = 0,
  chunks = 0,
}: HeaderProps) {
  return (
    <header className="top-header">
      <div className="header-main">
        <div className="header-title">
          <div className="header-icon">
            <ShieldCheck size={23} strokeWidth={2} />
          </div>

          <div className="header-copy">
            <h2>HSE Knowledge Assistant</h2>
            <p>
              Evidence-grounded industrial safety intelligence
            </p>
          </div>
        </div>

        <div
          className={`api-status ${
            isOnline ? "online" : "offline"
          }`}
        >
          {isOnline ? (
            <Wifi size={15} />
          ) : (
            <WifiOff size={15} />
          )}

          <span>
            {isOnline ? "API Online" : "API Offline"}
          </span>
        </div>
      </div>

      <div className="header-metrics">
        <div className="metric-item">
          <div className="metric-icon">
            <FileText size={17} />
          </div>

          <div className="metric-content">
            <span>Documents</span>
            <strong>{documents}</strong>
          </div>
        </div>

        <div className="metric-divider" />

        <div className="metric-item">
          <div className="metric-icon">
            <Database size={17} />
          </div>

          <div className="metric-content">
            <span>Knowledge Chunks</span>
            <strong>{chunks}</strong>
          </div>
        </div>

        <div className="metric-divider" />

        <div className="metric-item">
          <div className="metric-icon active">
            <Activity size={17} />
          </div>

          <div className="metric-content">
            <span>RAG Engine</span>

            <div className="rag-status">
              <span className="status-dot" />
              <strong>Active</strong>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;