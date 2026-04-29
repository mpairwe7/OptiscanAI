"use client";
import { useQuery } from "@tanstack/react-query";
import { fetchSystemInfo, fetchModelHealth, fetchKnowledgeGraph, fetchAgentStatus, fetchAgentEvents } from "@/lib/api";
import { BenchmarkPanel } from "@/components/benchmark-panel";

function InfoRow({ label, value, badge }: { label: string; value: string | number | null; badge?: "green" | "amber" | "red" }) {
  const badgeColors = { green: "bg-emerald-100 text-emerald-700", amber: "bg-amber-100 text-amber-700", red: "bg-red-100 text-red-700" };
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-50 last:border-0">
      <span className="text-sm text-slate-500">{label}</span>
      {badge ? (
        <span className={`text-xs font-medium px-2.5 py-0.5 rounded-full ${badgeColors[badge]}`}>
          {String(value)}
        </span>
      ) : (
        <span className="text-sm font-medium text-slate-700 font-mono">{String(value ?? "N/A")}</span>
      )}
    </div>
  );
}

export function SystemPage() {
  const sysInfo = useQuery({ queryKey: ["system-info"], queryFn: fetchSystemInfo, staleTime: 60_000 });
  const kg = useQuery({ queryKey: ["knowledge-graph"], queryFn: fetchKnowledgeGraph, staleTime: 300_000 });
  const agentStatus = useQuery({ queryKey: ["agent-status"], queryFn: fetchAgentStatus, refetchInterval: 10_000 });
  const agentEvents = useQuery({ queryKey: ["agent-events"], queryFn: () => fetchAgentEvents(15), refetchInterval: 10_000 });

  const sys = sysInfo.data;

  return (
    <div className="space-y-4 sm:space-y-6 animate-fade-in">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-800">System Status</h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-0.5">Infrastructure, model, and compliance details</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        {/* Platform Info */}
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100">
            <h2 className="font-semibold text-sm sm:text-base text-slate-800">Platform</h2>
          </div>
          <div className="px-4 sm:px-5 py-3">
            {sys ? (
              <>
                <InfoRow label="Name" value={sys.platform.name} />
                <InfoRow label="Version" value={sys.platform.version} />
                <InfoRow label="Environment" value={sys.platform.environment}
                         badge={sys.platform.environment === "production" ? "green" : "amber"} />
                <InfoRow label="Regulatory Mode" value={sys.platform.regulatory_mode}
                         badge={sys.platform.regulatory_mode === "research" ? "amber" : "green"} />
              </>
            ) : (
              <div className="py-4 text-sm text-slate-400 text-center">Loading...</div>
            )}
          </div>
        </div>

        {/* Infrastructure */}
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100">
            <h2 className="font-semibold text-sm sm:text-base text-slate-800">Infrastructure</h2>
          </div>
          <div className="px-4 sm:px-5 py-3">
            {sys ? (
              <>
                <InfoRow label="Python" value={sys.infrastructure.python_version} />
                <InfoRow label="PyTorch" value={sys.infrastructure.pytorch_version} />
                <InfoRow label="CUDA" value={sys.infrastructure.cuda_available ? sys.infrastructure.cuda_version : "CPU Only"}
                         badge={sys.infrastructure.cuda_available ? "green" : "amber"} />
                <InfoRow label="GPU" value={sys.infrastructure.gpu ?? "None"} />
                <InfoRow label="GPU Memory" value={sys.infrastructure.gpu_memory ?? "N/A"} />
                <InfoRow label="Device" value={sys.infrastructure.device} />
              </>
            ) : (
              <div className="py-4 text-sm text-slate-400 text-center">Loading...</div>
            )}
          </div>
        </div>

        {/* Model Details */}
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100">
            <h2 className="font-semibold text-sm sm:text-base text-slate-800">Model</h2>
          </div>
          <div className="px-4 sm:px-5 py-3">
            {sys ? (
              <>
                <InfoRow label="Architecture" value={sys.model.name.toUpperCase()} />
                <InfoRow label="Status" value={sys.model.loaded ? "Loaded" : "Demo Mode"}
                         badge={sys.model.loaded ? "green" : "amber"} />
                <InfoRow label="Disease Classes" value={sys.model.diseases_covered} />
                <InfoRow label="KG Relationships" value={sys.model.knowledge_graph_edges} />
                <InfoRow label="Threshold Mode" value={sys.model.threshold_source} />
              </>
            ) : (
              <div className="py-4 text-sm text-slate-400 text-center">Loading...</div>
            )}
          </div>
        </div>

        {/* Compliance */}
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100">
            <h2 className="font-semibold text-sm sm:text-base text-slate-800">Regulatory Compliance</h2>
          </div>
          <div className="px-4 sm:px-5 py-3">
            {sys ? (
              <>
                <InfoRow label="EU AI Act (2024/1689)" value={sys.compliance.eu_ai_act.replace(/_/g, " ")}
                         badge={sys.compliance.eu_ai_act === "conformity_ready" ? "green" : "amber"} />
                <InfoRow label="FDA SaMD" value={sys.compliance.fda_samd.replace(/_/g, " ")}
                         badge="amber" />
                <InfoRow label="Data Governance" value={sys.compliance.data_governance ? "Active" : "Inactive"}
                         badge={sys.compliance.data_governance ? "green" : "red"} />
                <InfoRow label="Model Cards" value={sys.compliance.model_cards ? "Active" : "Inactive"}
                         badge={sys.compliance.model_cards ? "green" : "red"} />
                <InfoRow label="Fairness Evaluation" value={sys.compliance.fairness_evaluation ? "Active" : "Inactive"}
                         badge={sys.compliance.fairness_evaluation ? "green" : "red"} />
                <InfoRow label="Audit Trail" value={sys.compliance.prediction_logging ? "Active" : "Inactive"}
                         badge={sys.compliance.prediction_logging ? "green" : "red"} />
              </>
            ) : (
              <div className="py-4 text-sm text-slate-400 text-center">Loading...</div>
            )}
          </div>
        </div>
      </div>

      {/* Autonomous Agents */}
      {agentStatus.data && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100 flex items-center justify-between">
            <h2 className="font-semibold text-sm sm:text-base text-slate-800">Autonomous Agents</h2>
            <span className="text-[10px] sm:text-xs text-slate-400 font-mono">
              {agentStatus.data.event_bus.total_events} events
            </span>
          </div>
          <div className="p-4 sm:p-5 space-y-3">
            {Object.entries(agentStatus.data.agents).map(([key, agent]) => {
              const statusColor = agent.status === "running" ? "bg-emerald-400" : agent.status === "error" ? "bg-red-400" : "bg-slate-400";
              return (
                <div key={key} className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg border border-slate-100">
                  <span className={`w-2.5 h-2.5 rounded-full mt-1.5 shrink-0 ${statusColor} ${agent.status === "running" ? "animate-pulse-dot" : ""}`} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-medium text-slate-700">{agent.name}</span>
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                        agent.status === "running" ? "bg-emerald-100 text-emerald-700" :
                        agent.status === "error" ? "bg-red-100 text-red-700" :
                        "bg-slate-100 text-slate-600"
                      }`}>{agent.status}</span>
                    </div>
                    <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1 text-[10px] text-slate-500">
                      <span>Actions: <strong className="text-slate-600">{agent.actions_taken}</strong></span>
                      <span>Errors: <strong className={agent.errors > 0 ? "text-red-600" : "text-slate-600"}>{agent.errors}</strong></span>
                      {agent.last_action && <span>Last: {agent.last_action}</span>}
                    </div>
                    <div className="flex flex-wrap gap-1 mt-1.5">
                      {agent.tools.map((tool) => (
                        <span key={tool} className="text-[9px] bg-white border border-slate-200 text-slate-500 px-1.5 py-0.5 rounded">
                          {tool}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Agent Event Stream */}
      {agentEvents.data && agentEvents.data.events.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100">
            <h2 className="font-semibold text-sm sm:text-base text-slate-800">Agent Event Stream</h2>
          </div>
          <div className="max-h-64 overflow-y-auto divide-y divide-slate-50">
            {agentEvents.data.events.map((evt) => (
              <div key={evt.event_id} className="px-4 sm:px-5 py-2 flex items-start gap-3">
                <span className="text-[9px] font-mono text-slate-400 w-14 shrink-0 pt-0.5">
                  {new Date(evt.timestamp).toLocaleTimeString()}
                </span>
                <span className="text-[10px] font-mono text-teal-600 w-32 shrink-0 truncate">{evt.type}</span>
                <span className="text-[10px] text-slate-500 truncate">{evt.source}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Knowledge Graph Summary */}
      {kg.data && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-slate-100">
            <h2 className="font-semibold text-sm sm:text-base text-slate-800">Clinical Knowledge Graph</h2>
          </div>
          <div className="p-4 sm:p-5">
            <div className="grid grid-cols-3 gap-4 sm:gap-6 mb-4">
              <div className="text-center">
                <div className="text-xl sm:text-2xl font-bold text-teal-700">{kg.data.diseases}</div>
                <div className="text-[10px] sm:text-xs text-slate-500">Diseases</div>
              </div>
              <div className="text-center">
                <div className="text-xl sm:text-2xl font-bold text-teal-700">{kg.data.edges}</div>
                <div className="text-[10px] sm:text-xs text-slate-500">Relationships</div>
              </div>
              <div className="text-center">
                <div className="text-xl sm:text-2xl font-bold text-teal-700">{Object.keys(kg.data.categories).length}</div>
                <div className="text-[10px] sm:text-xs text-slate-500">Categories</div>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(kg.data.categories).map(([cat, diseases]) => (
                <span key={cat} className="text-xs bg-slate-100 text-slate-600 px-2.5 py-1 rounded-full">
                  {cat.replace(/_/g, " ")} ({(diseases as string[]).length})
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Performance Benchmarks */}
      <div className="space-y-2">
        <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
          Live Performance Metrics
        </h2>
        <BenchmarkPanel />
      </div>
    </div>
  );
}
