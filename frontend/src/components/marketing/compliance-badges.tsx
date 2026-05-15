export function ComplianceBadges() {
  const badges = [
    { code: "EU AI ACT", caption: "Conformity-ready" },
    { code: "PDP 2019", caption: "Uganda data protection" },
    { code: "FHIR R5", caption: "Interoperable reports" },
    { code: "ISO 27001", caption: "Path to certification" },
    { code: "DICOM", caption: "Imaging interop" },
    { code: "EU MDR", caption: "Class IIa pathway" },
  ];
  return (
    <section className="py-12 bg-slate-900 text-white">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="text-center text-xs uppercase tracking-[0.18em] font-semibold text-slate-400">
          Built for clinical-grade compliance
        </div>
        <div className="mt-6 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {badges.map((b) => (
            <div
              key={b.code}
              className="rounded-xl border border-white/10 bg-white/5 px-3 py-3 text-center"
            >
              <div className="text-[11px] font-bold tracking-widest text-teal-400">{b.code}</div>
              <div className="mt-0.5 text-[10px] text-slate-400">{b.caption}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
