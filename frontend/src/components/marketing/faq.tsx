const FAQ_ITEMS = [
  {
    q: "Is OptiscanAI a medical device?",
    a: "OptiscanAI is currently classified as a screening decision-support tool. We're tracking EU MDR Class IIa and FDA SaMD conformity — talk to sales if your facility needs a regulatory dossier for procurement.",
  },
  {
    q: "How accurate is the model?",
    a: "Our 45-disease multi-label model reports per-disease AUC-ROC and F1 scores in the model card available on every Practice+ plan. We publish a fresh evaluation against held-out Ugandan-cohort data each release.",
  },
  {
    q: "Where is patient data stored?",
    a: "All scans are processed in-region (Africa) by default. The Health System tier supports on-prem deployment with no scan data leaving your network. We meet Uganda's PDP Act 2019 requirements.",
  },
  {
    q: "Can I cancel any time?",
    a: "Yes. Free downgrades happen instantly, Clinician and Practice cancellations take effect at the end of the current billing period — no refunds-required, no surprise fees. Manage everything from /app/billing.",
  },
  {
    q: "What if my clinic is offline?",
    a: "The Practice tier supports an offline-RAG sync bundle: ship the screening workflow to a tablet with a fundus camera, screen for a day, then sync when you're back on the network.",
  },
  {
    q: "Can I pay with mobile money?",
    a: "Yes — subscriptions are paid via MTN Mobile Money. A push prompt is sent to your phone; enter your MoMo PIN to confirm.",
  },
  {
    q: "Do you offer educational pricing?",
    a: "Yes — Makerere students get a Practice license at no cost as part of our research partnership. Email sales@makstartup.com from your institutional address.",
  },
];

export function FAQ() {
  return (
    <section id="faq" className="py-20 sm:py-24 bg-slate-50">
      <div className="max-w-3xl mx-auto px-4 sm:px-6">
        <div className="text-center">
          <div className="inline-flex items-center gap-1.5 text-xs font-semibold text-teal-700 bg-teal-50 px-2.5 py-1 rounded-full">
            FAQ
          </div>
          <h2 className="mt-4 text-3xl sm:text-4xl font-bold tracking-tight text-slate-900 text-balance">
            Common questions
          </h2>
        </div>
        <div className="mt-10 divide-y divide-slate-200 rounded-2xl border border-slate-200 bg-white">
          {FAQ_ITEMS.map((item) => (
            <details key={item.q} className="group p-5 sm:p-6">
              <summary className="cursor-pointer list-none flex items-start justify-between gap-3 font-semibold text-slate-900">
                <span>{item.q}</span>
                <svg
                  className="w-5 h-5 text-slate-400 shrink-0 mt-0.5 transition-transform group-open:rotate-180"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                  aria-hidden
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                </svg>
              </summary>
              <p className="mt-3 text-sm text-slate-600 leading-relaxed">{item.a}</p>
            </details>
          ))}
        </div>

        {/* JSON-LD for SEO */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "FAQPage",
              mainEntity: FAQ_ITEMS.map((i) => ({
                "@type": "Question",
                name: i.q,
                acceptedAnswer: { "@type": "Answer", text: i.a },
              })),
            }),
          }}
        />
      </div>
    </section>
  );
}
