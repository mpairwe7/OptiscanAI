export const metadata = { title: "Contact sales — OptiscanAI" };

const SALES_EMAIL = "mpairwelauben25@gmail.com";
// Uganda E.164 format for wa.me deep link: 256 + 773336896 (drop leading 0).
const WHATSAPP_NUMBER = "256773336896";
const WHATSAPP_DISPLAY = "+256 773 336 896";
const WHATSAPP_PREFILL =
  "Hi OptiscanAI team — I'd like to talk about Health System pricing.";

export default function ContactSalesPage() {
  const waLink = `https://wa.me/${WHATSAPP_NUMBER}?text=${encodeURIComponent(WHATSAPP_PREFILL)}`;

  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-6 py-10 sm:py-16">
      <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-slate-900 text-balance">
        Health System pricing — let&apos;s talk
      </h1>
      <p className="mt-3 text-slate-600 text-pretty">
        Unlimited scans, SSO + SCIM, DHIS2 / FHIR / DICOM integrations, BAA + Uganda PDP Act
        compliance, dedicated CSM + SLA. Tell us about your facility and we&apos;ll get back within
        one business day.
      </p>

      {/* Fast-path: WhatsApp + email buttons before the form */}
      <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3">
        <a
          href={waLink}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`Chat with sales on WhatsApp at ${WHATSAPP_DISPLAY}`}
          className="inline-flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-[#25D366] hover:bg-[#1ebe57] text-white font-semibold text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#25D366] focus-visible:ring-offset-2 min-h-[44px]"
        >
          {/* WhatsApp icon */}
          <svg
            className="w-5 h-5"
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden="true"
            focusable="false"
          >
            <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51l-.57-.01c-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.626.712.226 1.36.194 1.872.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z" />
          </svg>
          <span>WhatsApp {WHATSAPP_DISPLAY}</span>
        </a>
        <a
          href={`mailto:${SALES_EMAIL}`}
          aria-label={`Email sales at ${SALES_EMAIL}`}
          className="inline-flex items-center justify-center gap-2 px-4 py-3 rounded-lg border border-slate-300 hover:bg-slate-50 text-slate-700 font-semibold text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2 min-h-[44px]"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.8}
            aria-hidden="true"
            focusable="false"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 8l9 6 9-6M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
            />
          </svg>
          <span className="truncate">{SALES_EMAIL}</span>
        </a>
      </div>

      <div className="mt-8 mb-3 flex items-center gap-3 text-xs uppercase tracking-wider font-semibold text-slate-400">
        <div className="flex-1 h-px bg-slate-200" />
        <span>or send us the details</span>
        <div className="flex-1 h-px bg-slate-200" />
      </div>

      <form
        action={`mailto:${SALES_EMAIL}`}
        method="post"
        encType="text/plain"
        className="space-y-4 rounded-xl border border-slate-200 bg-white p-5 sm:p-6"
        aria-label="Contact sales form"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label htmlFor="cs-name" className="block text-sm font-medium text-slate-700">
              Full name
            </label>
            <input
              id="cs-name"
              name="name"
              type="text"
              autoComplete="name"
              required
              aria-required="true"
              className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 focus-visible:outline-none min-h-[44px]"
            />
          </div>
          <div>
            <label htmlFor="cs-email" className="block text-sm font-medium text-slate-700">
              Work email
            </label>
            <input
              id="cs-email"
              type="email"
              name="email"
              autoComplete="email"
              inputMode="email"
              required
              aria-required="true"
              className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 focus-visible:outline-none min-h-[44px]"
            />
          </div>
        </div>
        <div>
          <label htmlFor="cs-facility" className="block text-sm font-medium text-slate-700">
            Facility / organisation
          </label>
          <input
            id="cs-facility"
            name="facility"
            type="text"
            autoComplete="organization"
            required
            aria-required="true"
            className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 focus-visible:outline-none min-h-[44px]"
          />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label htmlFor="cs-country" className="block text-sm font-medium text-slate-700">
              Country
            </label>
            <input
              id="cs-country"
              name="country"
              type="text"
              autoComplete="country-name"
              defaultValue="Uganda"
              className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 focus-visible:outline-none min-h-[44px]"
            />
          </div>
          <div>
            <label htmlFor="cs-volume" className="block text-sm font-medium text-slate-700">
              Approx. monthly scan volume
            </label>
            <select
              id="cs-volume"
              name="volume"
              className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 focus-visible:outline-none min-h-[44px]"
            >
              <option>&lt; 5,000</option>
              <option>5,000 – 20,000</option>
              <option>20,000 – 100,000</option>
              <option>&gt; 100,000</option>
            </select>
          </div>
        </div>
        <div>
          <label htmlFor="cs-message" className="block text-sm font-medium text-slate-700">
            Anything else?
          </label>
          <textarea
            id="cs-message"
            name="message"
            rows={4}
            aria-describedby="cs-message-hint"
            className="mt-1 w-full px-3 py-2 rounded-lg border border-slate-300 focus:border-teal-500 focus:ring-1 focus:ring-teal-500 focus-visible:outline-none"
          />
          <p id="cs-message-hint" className="mt-1 text-xs text-slate-500">
            Mention any specific integrations (DHIS2, FHIR, DICOM) or compliance asks.
          </p>
        </div>
        <button
          type="submit"
          className="w-full inline-flex items-center justify-center px-4 py-3 text-sm font-semibold rounded-lg bg-teal-600 hover:bg-teal-700 text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2 min-h-[44px]"
        >
          Request a demo
        </button>
        <p className="text-xs text-slate-500 text-center">
          By submitting, you agree to our{" "}
          <a href="/legal/privacy" className="underline hover:text-slate-700">
            privacy policy
          </a>
          .
        </p>
      </form>
    </div>
  );
}
