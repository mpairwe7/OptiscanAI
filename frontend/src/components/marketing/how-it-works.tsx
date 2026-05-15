export function HowItWorks() {
  const steps = [
    {
      n: "01",
      title: "Upload a fundus image",
      body: "JPEG or PNG, captured on any handheld or table-top fundus camera. The fundus-gate v2 model rejects non-fundus images in 12 ms.",
      visual: (
        <div className="aspect-[4/3] rounded-xl border border-dashed border-slate-300 bg-slate-50 flex items-center justify-center text-slate-400">
          <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
        </div>
      ),
    },
    {
      n: "02",
      title: "AI analyzes 45 diseases",
      body: "Multi-label inference + Grad-CAM heatmaps + clinical knowledge-graph refinement. Median end-to-end latency 85 ms.",
      visual: (
        <div className="aspect-[4/3] rounded-xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4 flex flex-col gap-2 text-white">
          <div className="text-[10px] uppercase tracking-wider font-bold text-teal-400">Predictions</div>
          {[
            { label: "Diabetic Retinopathy", pct: 94, color: "bg-red-500" },
            { label: "Hypertensive Retinopathy", pct: 68, color: "bg-amber-500" },
            { label: "Macular Edema", pct: 41, color: "bg-yellow-500" },
            { label: "Glaucoma Suspect", pct: 22, color: "bg-emerald-500" },
          ].map((p) => (
            <div key={p.label}>
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium">{p.label}</span>
                <span className="font-mono opacity-70">{p.pct}%</span>
              </div>
              <div className="mt-0.5 h-1 bg-white/10 rounded-full overflow-hidden">
                <div className={`h-full ${p.color}`} style={{ width: `${p.pct}%` }} />
              </div>
            </div>
          ))}
        </div>
      ),
    },
    {
      n: "03",
      title: "Refer with confidence",
      body: "PDF report, FHIR DiagnosticReport, SMS referral to the patient. DHIS2 push for Health System tier — everything an EMR needs.",
      visual: (
        <div className="aspect-[4/3] rounded-xl bg-white border border-slate-200 p-4 flex flex-col gap-2 shadow-inner">
          <div className="text-[10px] uppercase tracking-wider font-bold text-slate-500">Referral</div>
          <div className="rounded-lg bg-red-50 border border-red-200 p-2 text-xs">
            <div className="font-semibold text-red-700">HIGH priority</div>
            <div className="text-red-600 mt-0.5">Refer to ophthalmology within 2 weeks</div>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-600">
            <span className="px-2 py-0.5 rounded bg-slate-100 font-mono">FHIR</span>
            <span className="px-2 py-0.5 rounded bg-slate-100 font-mono">DHIS2</span>
            <span className="px-2 py-0.5 rounded bg-slate-100 font-mono">SMS</span>
          </div>
          <div className="text-xs text-slate-500 mt-auto">Patient: +256 700 ••• 234</div>
        </div>
      ),
    },
  ];

  return (
    <section id="how-it-works" className="py-20 sm:py-28">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-2xl mx-auto">
          <div className="inline-flex items-center gap-1.5 text-xs font-semibold text-teal-700 bg-teal-50 px-2.5 py-1 rounded-full">
            How it works
          </div>
          <h2 className="mt-4 text-3xl sm:text-4xl font-bold tracking-tight text-slate-900 text-balance">
            From fundus image to clinical decision in three steps
          </h2>
          <p className="mt-3 text-slate-600 text-pretty">
            Every prediction comes with the visual evidence and clinical context a clinician needs to act.
          </p>
        </div>

        <div className="mt-14 grid gap-6 lg:grid-cols-3 lg:gap-8">
          {steps.map((s, i) => (
            <div key={s.n} className="relative">
              <div className="text-xs font-mono font-bold text-teal-600 tracking-widest">{s.n}</div>
              <h3 className="mt-2 text-xl font-semibold text-slate-900">{s.title}</h3>
              <p className="mt-1.5 text-sm text-slate-600">{s.body}</p>
              <div className="mt-4">{s.visual}</div>

              {/* Connector arrow on desktop */}
              {i < steps.length - 1 && (
                <svg
                  className="hidden lg:block absolute top-2 -right-5 w-10 h-10 text-slate-300"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                  aria-hidden
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 5l7 7-7 7M5 12h15" />
                </svg>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
