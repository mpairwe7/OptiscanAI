export const metadata = { title: "Contact sales — OptiscanAI" };

export default function ContactSalesPage() {
  return (
    <div className="max-w-2xl mx-auto px-4 py-12 sm:py-16">
      <h1 className="text-4xl font-bold tracking-tight text-slate-900 text-balance">
        Health System pricing — let&apos;s talk
      </h1>
      <p className="mt-3 text-slate-600 text-pretty">
        Unlimited scans, SSO + SCIM, DHIS2 / FHIR / DICOM integrations, BAA + Uganda PDP Act compliance,
        dedicated CSM + SLA. Tell us about your facility and we&apos;ll get back within one business day.
      </p>

      <form
        action="mailto:sales@makstartup.com"
        method="post"
        encType="text/plain"
        className="mt-8 space-y-4 rounded-xl border border-slate-200 bg-white p-6"
      >
        <div className="grid sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700">Full name</label>
            <input
              name="name"
              required
              className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700">Work email</label>
            <input
              type="email"
              name="email"
              required
              className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            />
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700">Facility / organisation</label>
          <input
            name="facility"
            required
            className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700">Country</label>
          <input
            name="country"
            defaultValue="Uganda"
            className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700">Approx. monthly scan volume</label>
          <select
            name="volume"
            className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
          >
            <option>&lt; 5,000</option>
            <option>5,000 – 20,000</option>
            <option>20,000 – 100,000</option>
            <option>&gt; 100,000</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700">Anything else?</label>
          <textarea
            name="message"
            rows={4}
            className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
          />
        </div>
        <button
          type="submit"
          className="w-full inline-flex items-center justify-center px-4 py-2.5 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white"
        >
          Request a demo
        </button>
        <p className="text-xs text-slate-500 text-center">
          Or email <a href="mailto:sales@makstartup.com" className="underline">sales@makstartup.com</a> directly.
        </p>
      </form>
    </div>
  );
}
