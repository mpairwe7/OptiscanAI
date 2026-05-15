export function Testimonials() {
  const items = [
    {
      quote:
        "The Grad-CAM overlays let me show a registrar exactly which retinal region drove the diabetic-retinopathy alert. It's the first AI tool I'd actually defend in a case review.",
      name: "Dr Sarah K.",
      role: "Ophthalmologist · Mulago Hospital",
      initials: "SK",
      color: "bg-teal-500",
    },
    {
      quote:
        "We process 200+ scans per outreach day. The 85 ms latency means we don't queue patients, and the FHIR export drops straight into our EMR.",
      name: "James M.",
      role: "Practice manager · Mengo Eye Clinic",
      initials: "JM",
      color: "bg-indigo-500",
    },
    {
      quote:
        "The audit log + fairness dashboard is what convinced our IRB. We're now using OptiscanAI as the screening backbone for our diabetic-retinopathy outreach study.",
      name: "Prof. R. Ndegwa",
      role: "Researcher · Makerere University",
      initials: "RN",
      color: "bg-amber-500",
    },
  ];

  return (
    <section className="py-20 sm:py-24">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-2xl mx-auto">
          <div className="inline-flex items-center gap-1.5 text-xs font-semibold text-teal-700 bg-teal-50 px-2.5 py-1 rounded-full">
            What clinicians say
          </div>
          <h2 className="mt-4 text-3xl sm:text-4xl font-bold tracking-tight text-slate-900 text-balance">
            Built with the clinicians using it every day
          </h2>
        </div>

        <div className="mt-12 grid gap-5 md:grid-cols-3">
          {items.map((t) => (
            <figure
              key={t.name}
              className="rounded-2xl border border-slate-200 bg-white p-6 flex flex-col"
            >
              <svg className="w-8 h-8 text-teal-200" fill="currentColor" viewBox="0 0 32 32" aria-hidden>
                <path d="M9.352 4C4.456 7.456 1 13.12 1 19.36 1 24.832 4.736 28 8.96 28c3.84 0 6.624-3.072 6.624-6.624 0-3.456-2.4-5.952-5.376-5.952-.576 0-1.344.096-1.536.192.48-3.168 3.744-6.912 7.008-8.832L9.352 4zm15.328 0c-4.8 3.456-8.256 9.12-8.256 15.36 0 5.472 3.744 8.64 7.968 8.64 3.84 0 6.624-3.072 6.624-6.624 0-3.456-2.4-5.952-5.376-5.952-.576 0-1.344.096-1.536.192.48-3.168 3.744-6.912 7.008-8.832L24.68 4z" />
              </svg>
              <blockquote className="mt-3 text-slate-700 text-sm leading-relaxed flex-1">
                &ldquo;{t.quote}&rdquo;
              </blockquote>
              <figcaption className="mt-5 flex items-center gap-3">
                <div className={`w-10 h-10 rounded-full ${t.color} text-white text-sm font-bold flex items-center justify-center`}>
                  {t.initials}
                </div>
                <div>
                  <div className="font-semibold text-slate-900 text-sm">{t.name}</div>
                  <div className="text-xs text-slate-500">{t.role}</div>
                </div>
              </figcaption>
            </figure>
          ))}
        </div>
      </div>
    </section>
  );
}
