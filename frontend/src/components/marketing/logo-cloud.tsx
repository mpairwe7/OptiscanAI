import { PartnerLogo, type PartnerKey } from "./partner-logo";

interface Group {
  label: string;
  partners: PartnerKey[];
}

const GROUPS: Group[] = [
  {
    label: "Built with",
    partners: ["makerere", "makstartup"],
  },
  {
    label: "Pilot facilities",
    partners: ["mulago", "ruharo", "mengo", "lubaga"],
  },
  {
    label: "Research & infrastructure partners",
    partners: ["idi", "agakhan", "moh-uganda", "crane-cloud"],
  },
];

export function LogoCloud() {
  return (
    <section className="py-14 sm:py-16 border-y border-slate-200 bg-white">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <p className="text-center text-[11px] uppercase tracking-[0.22em] font-semibold text-slate-500">
          Trusted by clinicians and researchers across Uganda
        </p>

        <div className="mt-10 space-y-7">
          {GROUPS.map((g) => (
            <div key={g.label}>
              <div className="text-[10px] uppercase tracking-widest text-slate-400 font-bold text-center mb-3">
                {g.label}
              </div>
              <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-4 sm:gap-x-12 text-slate-500 hover:text-slate-700 transition-colors">
                {g.partners.map((p) => (
                  <PartnerLogo
                    key={p}
                    name={p}
                    className="opacity-70 hover:opacity-100 transition-opacity"
                  />
                ))}
              </div>
            </div>
          ))}
        </div>

        <p className="mt-10 text-center text-[11px] text-slate-400">
          Logos shown represent active and prospective pilot partners.
          <span className="hidden sm:inline">
            {" "}
            Reach out via{" "}
            <a href="/contact-sales" className="underline hover:text-slate-700">
              contact sales
            </a>{" "}
            to add your facility.
          </span>
        </p>
      </div>
    </section>
  );
}
