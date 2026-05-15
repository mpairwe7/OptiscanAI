import Link from "next/link";
import { LegalDoc, type LegalSection } from "@/components/marketing/legal-doc";

export const metadata = {
  title: "Terms of Service",
  description:
    "Terms governing your use of OptiscanAI — a clinical retinal-screening decision-support service operated by MakStartup.",
};

const SECTIONS: LegalSection[] = [
  {
    id: "acceptance",
    title: "Acceptance of these terms",
    body: (
      <>
        <p>
          By creating an account on{" "}
          <Link href="/" className="text-teal-700 underline">www.optiscan.makstartup.com</Link>{" "}
          you (the &ldquo;<strong>Customer</strong>&rdquo;) agree to these Terms of Service and the
          accompanying{" "}
          <Link href="/legal/privacy" className="text-teal-700 underline">Privacy Policy</Link>.
          If you are signing up on behalf of an organisation, you confirm that you have authority to
          bind that organisation to these terms.
        </p>
      </>
    ),
  },
  {
    id: "service",
    title: "What OptiscanAI is",
    body: (
      <>
        <p>
          OptiscanAI is an AI-powered retinal screening platform that produces multi-label predictions,
          explainability artefacts (Grad-CAM, SHAP, LIME, Integrated Gradients, ELI5), and clinical
          knowledge-graph reasoning for fundus images uploaded by the Customer. The service is
          provided over the public internet on a software-as-a-service basis.
        </p>
        <p>
          <strong>OptiscanAI is decision-support, not a diagnostic device.</strong> Predictions must
          be reviewed by a licensed clinician before any clinical action. The service is not a
          substitute for clinical examination and is not currently CE-marked or FDA-cleared as a
          medical device.
        </p>
      </>
    ),
  },
  {
    id: "eligibility",
    title: "Who can use the service",
    body: (
      <>
        <p>
          Paid tiers (Clinician, Practice, Health System) are available to licensed healthcare
          practitioners, healthcare organisations, and academic researchers. The Free tier is open
          to all, subject to the acceptable-use clause.
        </p>
      </>
    ),
  },
  {
    id: "accounts",
    title: "Account responsibilities",
    body: (
      <>
        <p>
          You are responsible for keeping your password and refresh tokens confidential, for all
          activity that happens under your account, and for keeping your contact email current. If
          you suspect unauthorised use, sign out of all sessions from{" "}
          <Link href="/app/account" className="text-teal-700 underline">/app/account</Link> and
          contact{" "}
          <a href="mailto:support@makstartup.com" className="text-teal-700 underline">support</a>{" "}
          immediately.
        </p>
      </>
    ),
  },
  {
    id: "acceptable-use",
    title: "Acceptable use",
    body: (
      <>
        <p>You may not, and may not allow any user of your organisation to:</p>
        <ul className="list-disc pl-5 space-y-1.5">
          <li>Upload imagery you are not authorised to share, or that you have not obtained appropriate patient consent for under the PDP Act 2019</li>
          <li>Use the service to make a binding clinical diagnosis without practitioner review</li>
          <li>Probe, scan, or test the vulnerability of the service without prior written authorisation</li>
          <li>Reverse engineer, decompile, or attempt to extract the underlying models</li>
          <li>Resell, sublicense, or white-label the service without a separate written agreement</li>
        </ul>
      </>
    ),
  },
  {
    id: "billing",
    title: "Subscriptions, billing, refunds",
    body: (
      <>
        <p>
          Paid plans bill in advance, monthly or annually as selected at checkout, in US dollars.
          Mobile-money charges are converted to UGX at the FX rate displayed at checkout time. You
          may cancel at any time from{" "}
          <Link href="/app/billing" className="text-teal-700 underline">/app/billing</Link>; access
          continues until the end of the current period. We don&apos;t offer prorated refunds for
          early cancellation, but we will refund duplicate or accidental charges within 30 days.
        </p>
        <p>
          Free tier accounts that exceed the published quota (10 scans/month) will receive a
          paywall prompt; we will never silently charge a Free account.
        </p>
      </>
    ),
  },
  {
    id: "ip",
    title: "Intellectual property",
    body: (
      <>
        <p>
          You retain all rights to the fundus images you upload and the patient information you
          enter. We retain all rights to the OptiscanAI service, including the models, knowledge
          graph, software, and documentation. You receive a non-exclusive, non-transferable licence
          to use the service for the duration of your subscription.
        </p>
        <p>
          Aggregated, de-identified usage statistics may be used by us to improve the service and to
          publish research, provided no individual patient can be re-identified.
        </p>
      </>
    ),
  },
  {
    id: "clinical-disclaimer",
    title: "Clinical decision-support disclaimer",
    body: (
      <>
        <p>
          OptiscanAI&apos;s predictions reflect statistical patterns learned from training data and
          will occasionally be wrong. A &ldquo;HIGH priority&rdquo; referral does not guarantee
          disease presence and a &ldquo;LOW priority&rdquo; output does not guarantee its absence.
          The practitioner must apply independent clinical judgement, take a history, and where
          relevant perform a slit-lamp examination before issuing a diagnosis or treatment plan.
        </p>
      </>
    ),
  },
  {
    id: "availability",
    title: "Service availability",
    body: (
      <>
        <p>
          We target 99.5% monthly uptime for paid plans, measured at the public ingress. Planned
          maintenance is announced 48 hours in advance on the system status footer. Health System
          contracts may include a stronger SLA negotiated in writing.
        </p>
      </>
    ),
  },
  {
    id: "termination",
    title: "Termination",
    body: (
      <>
        <p>
          You may terminate your account at any time by cancelling from{" "}
          <Link href="/app/billing" className="text-teal-700 underline">/app/billing</Link>. We may
          suspend or terminate accounts that breach the acceptable-use clause, fail to pay, or pose
          a credible safety risk to patients. On termination we delete personal data within 30 days,
          retaining only the audit metadata required by clinical-governance regulation.
        </p>
      </>
    ),
  },
  {
    id: "liability",
    title: "Limitation of liability",
    body: (
      <>
        <p>
          To the maximum extent permitted by Ugandan law, our aggregate liability arising from or
          related to your use of the service in any twelve-month period is limited to the fees you
          paid us in that same period. We are not liable for indirect, incidental, special, or
          consequential damages, including loss of profits, revenue, data, or business opportunity.
        </p>
        <p>
          Nothing in this clause limits liability that cannot lawfully be limited (for example,
          liability for gross negligence or fraud).
        </p>
      </>
    ),
  },
  {
    id: "indemnity",
    title: "Indemnity",
    body: (
      <>
        <p>
          You agree to indemnify us against any third-party claim arising from your or your
          organisation&apos;s misuse of the service, breach of the acceptable-use clause, or upload
          of imagery without lawful basis.
        </p>
      </>
    ),
  },
  {
    id: "governing-law",
    title: "Governing law and disputes",
    body: (
      <>
        <p>
          These terms are governed by the laws of the Republic of Uganda. Any dispute arising from
          them will first be addressed in good-faith negotiation; failing resolution within 30 days,
          it will be referred to arbitration under the Arbitration and Conciliation Act, seat
          Kampala, language English.
        </p>
      </>
    ),
  },
  {
    id: "changes",
    title: "Changes to these terms",
    body: (
      <p>
        We will post material changes to this page at least 14 days before they take effect and
        notify account owners by email. Continued use of the service after the effective date
        constitutes acceptance.
      </p>
    ),
  },
];

export default function TermsPage() {
  return (
    <LegalDoc
      title="Terms of Service"
      subtitle="The legal agreement that governs your use of OptiscanAI."
      lastUpdated="2026-05-15"
      effectiveDate="2026-05-15"
      sections={SECTIONS}
      contact={
        <p>
          Questions about these terms? Email{" "}
          <a href="mailto:legal@makstartup.com" className="text-teal-700 underline">
            legal@makstartup.com
          </a>{" "}
          or write to MakStartup, Plot 51, Makerere Hill Road, Kampala, Uganda.
        </p>
      }
    />
  );
}
