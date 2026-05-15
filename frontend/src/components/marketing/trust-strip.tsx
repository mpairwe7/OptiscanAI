export function TrustStrip() {
  const items = [
    { label: "Diseases detected", value: "45" },
    { label: "Explainability methods", value: "5" },
    { label: "EU AI Act", value: "Conformity-ready" },
    { label: "Locale", value: "🇺🇬 en-UG" },
  ];
  return (
    <section className="border-y border-slate-200 bg-slate-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6 grid grid-cols-2 sm:grid-cols-4 gap-6">
        {items.map((i) => (
          <div key={i.label} className="text-center">
            <div className="text-2xl font-bold text-slate-900">{i.value}</div>
            <div className="text-xs text-slate-500 mt-0.5">{i.label}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
