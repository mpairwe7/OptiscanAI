"use client";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useEffect, useState, type ReactNode } from "react";
import { apiMe } from "@/lib/auth-api";
import { useAuthStore } from "@/stores/auth-store";
import { PaywallModal } from "@/components/billing/paywall-modal";
import { UpsellSheet } from "@/components/billing/upsell-sheet";

function AuthHydrator() {
  const setUser = useAuthStore((s) => s.setUser);
  const setHydrated = useAuthStore((s) => s.setHydrated);

  const me = useQuery({
    queryKey: ["auth", "me"],
    queryFn: apiMe,
    retry: 0,
    staleTime: 60_000,
  });

  useEffect(() => {
    if (me.isSuccess) {
      setUser(me.data);
      setHydrated(true);
    } else if (me.isError) {
      setUser(null);
      setHydrated(true);
    }
  }, [me.isSuccess, me.isError, me.data, setUser, setHydrated]);

  return null;
}

export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30_000,
            gcTime: 120_000,
            retry: 1,
            refetchOnWindowFocus: false,
          },
        },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <AuthHydrator />
      {children}
      <PaywallModal />
      <UpsellSheet />
    </QueryClientProvider>
  );
}
