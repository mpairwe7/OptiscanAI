/**
 * Auth gate for /app/*.
 *
 * Note: this Next.js renames `middleware` → `proxy`. See
 *   node_modules/next/dist/docs/01-app/03-api-reference/03-file-conventions/proxy.md
 */
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function proxy(request: NextRequest) {
  const hasAccess = request.cookies.has("os_access");
  const hasRefresh = request.cookies.has("os_refresh");

  if (!hasAccess && !hasRefresh) {
    const url = request.nextUrl.clone();
    url.pathname = "/sign-in";
    url.searchParams.set("next", request.nextUrl.pathname + request.nextUrl.search);
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/app/:path*"],
};
