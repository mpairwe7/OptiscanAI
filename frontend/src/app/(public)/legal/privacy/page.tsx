import Link from "next/link";
import { LegalDoc, type LegalSection } from "@/components/marketing/legal-doc";

export const metadata = {
  title: "Privacy Policy",
  description:
    "How OptiscanAI collects, processes, and protects personal and clinical data — aligned with Uganda's PDP Act 2019.",
};

const SECTIONS: LegalSection[] = [
  {
    id: "overview",
    title: "Overview",
    body: (
      <>
        <p>
          OptiscanAI (&ldquo;<strong>we</strong>&rdquo;, &ldquo;<strong>us</strong>&rdquo;) is operated by
          MakStartup, a Uganda-based research and engineering team building clinical AI for African
          healthcare. This policy explains what personal and clinical information we collect when you
          use <Link href="/" className="text-teal-700 underline">www.optiscan.makstartup.com</Link>,
          why we collect it, and the rights you have over it.
        </p>
        <p>
          OptiscanAI is positioned as a screening decision-support tool. It does not replace a
          clinician&apos;s independent judgement and any output it produces must be reviewed by a
          qualified practitioner before a clinical action is taken.
        </p>
      </>
    ),
  },
  {
    id: "data-we-collect",
    title: "What data we collect",
    body: (
      <>
        <p>We collect only what is needed to operate the service. Specifically:</p>
        <ul className="list-disc pl-5 space-y-1.5">
          <li>
            <strong>Account data</strong> — your email address, full name, password hash (argon2),
            practitioner role, country, and the organisation you belong to.
          </li>
          <li>
            <strong>Billing data</strong> — invoices and payment metadata. Card numbers never touch
            our servers; they are processed by Stripe. Mobile-money phone numbers are stored hashed
            after a payment intent settles.
          </li>
          <li>
            <strong>Scan data</strong> — fundus images you upload, the predictions produced, and an
            audit record (timestamp, request ID, reviewer, referral priority).
          </li>
          <li>
            <strong>Operational data</strong> — IP address, browser/user-agent, and a rolling 60-day
            rate-limit bucket. We do not run third-party analytics on the marketing site.
          </li>
        </ul>
      </>
    ),
  },
  {
    id: "how-we-use",
    title: "How we use your data",
    body: (
      <>
        <p>
          Account and billing data are used to operate your subscription. Scan data is used to
          generate the prediction you requested and to maintain an immutable audit trail for clinical
          governance. Aggregated, de-identified scan metadata may be used to monitor model drift and
          fairness across protected attributes (age group, sex, geography, camera device).
        </p>
        <p>
          We do not sell personal data, and we do not use clinical data to train third-party models.
          Re-training of OptiscanAI models on patient data only happens when an organisation has
          opted in via a signed data-use agreement.
        </p>
      </>
    ),
  },
  {
    id: "phi",
    title: "Clinical / personal-health information",
    body: (
      <>
        <p>
          Fundus images and their derived predictions are treated as personal health information
          (PHI) under the Uganda PDP Act 2019. PHI is stored encrypted at rest, transferred over TLS,
          and accessed only by the clinicians and admins within the organisation the data belongs to.
        </p>
        <p>
          Health-System tier deployments can opt for fully on-premise inference — in that mode no
          fundus image leaves your facility. Talk to{" "}
          <Link href="/contact-sales" className="text-teal-700 underline">
            sales
          </Link>{" "}
          if your data-residency requirements demand this.
        </p>
      </>
    ),
  },
  {
    id: "residency",
    title: "Where your data lives",
    body: (
      <>
        <p>
          The default deployment runs in African data centres (Crane Cloud, Uganda). Backup snapshots
          are retained for 30 days. Cross-border transfers happen only when invoking explicitly opted-
          in third-party services — Stripe (United States), MTN MoMo (Uganda), Airtel Money (Uganda),
          Flutterwave (Africa-wide), and your email provider.
        </p>
      </>
    ),
  },
  {
    id: "subprocessors",
    title: "Sub-processors",
    body: (
      <>
        <p>The third parties that may handle your data on our behalf are limited to:</p>
        <ul className="list-disc pl-5 space-y-1.5">
          <li><strong>Stripe</strong> — card billing</li>
          <li><strong>MTN MoMo, Airtel Money, Flutterwave</strong> — mobile-money billing</li>
          <li><strong>Africa&apos;s Talking</strong> — SMS referral delivery (Health System tier)</li>
          <li><strong>Crane Cloud</strong> — primary cloud hosting (Uganda)</li>
          <li><strong>Anthropic, Groq</strong> — agentic-AI reasoning (Practice tier and above, opt-in)</li>
        </ul>
        <p>
          We will give 30 days&apos; notice on this page before adding any new sub-processor that
          handles PHI.
        </p>
      </>
    ),
  },
  {
    id: "rights",
    title: "Your rights under the PDP Act 2019",
    body: (
      <>
        <p>You can ask us at any time to:</p>
        <ul className="list-disc pl-5 space-y-1.5">
          <li>Tell you what data we hold about you</li>
          <li>Correct inaccurate data</li>
          <li>Delete your account and the personal data attached to it (we retain audit metadata required for clinical governance)</li>
          <li>Export your data in a portable format (CSV, FHIR Bundle)</li>
          <li>Withdraw consent to specific processing</li>
        </ul>
        <p>
          Email <a href="mailto:privacy@makstartup.com" className="text-teal-700 underline">privacy@makstartup.com</a>{" "}
          and we&apos;ll respond within 7 working days.
        </p>
      </>
    ),
  },
  {
    id: "cookies",
    title: "Cookies and session storage",
    body: (
      <>
        <p>
          We use two httpOnly cookies on the authenticated app: <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">os_access</code>{" "}
          (15-minute access token) and <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">os_refresh</code>{" "}
          (30-day rotating refresh token). Both are <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">SameSite=Lax</code>{" "}
          and <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">Secure</code> in production.
          We do not set tracking cookies on the marketing site.
        </p>
      </>
    ),
  },
  {
    id: "children",
    title: "Children",
    body: (
      <>
        <p>
          The OptiscanAI account product is intended for use by licensed healthcare practitioners
          and their staff. Children may, however, appear in scan data uploaded by a clinician for
          screening. We treat all such scans with the same encryption and access controls described
          above and require the uploading clinician to obtain appropriate consent under the PDP Act.
        </p>
      </>
    ),
  },
  {
    id: "breaches",
    title: "Data breaches",
    body: (
      <>
        <p>
          If we become aware of a breach that may affect your personal data, we will notify you and
          the Personal Data Protection Office of Uganda within 72 hours, in line with section 23 of
          the PDP Act. We will explain what happened, what data was involved, and what we are doing
          to mitigate the impact.
        </p>
      </>
    ),
  },
  {
    id: "changes",
    title: "Changes to this policy",
    body: (
      <p>
        We will post material changes to this page at least 14 days before they take effect, and
        notify account owners by email.
      </p>
    ),
  },
];

export default function PrivacyPage() {
  return (
    <LegalDoc
      title="Privacy Policy"
      subtitle="How we collect, process, and protect personal and clinical data."
      lastUpdated="2026-05-15"
      effectiveDate="2026-05-15"
      sections={SECTIONS}
      contact={
        <p>
          Questions about this policy? Email{" "}
          <a href="mailto:privacy@makstartup.com" className="text-teal-700 underline">
            privacy@makstartup.com
          </a>{" "}
          or write to MakStartup, Plot 51, Makerere Hill Road, Kampala, Uganda.
        </p>
      }
    />
  );
}
