export function LogoCloud() {
  // Placeholder partners — replace with real logos as pilots are signed.
  const partners = [
    "Makerere University",
    "Mulago Hospital",
    "Mengo Hospital",
    "Lubaga Hospital",
    "Ruharo Eye Hospital",
    "MakStartup",
  ];

  return (
    <section className="py-12 sm:py-16 border-y border-slate-200 bg-white">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="text-center text-xs uppercase tracking-[0.18em] font-semibold text-slate-500">
          Trusted by clinicians and researchers across Uganda
        </div>
        <div className="mt-8 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-x-6 gap-y-5 items-center">
          {partners.map((p) => (
            <div
              key={p}
              className="text-center text-sm font-semibold tracking-tight text-slate-400 hover:text-slate-700 transition-colors"
            >
              {p}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
