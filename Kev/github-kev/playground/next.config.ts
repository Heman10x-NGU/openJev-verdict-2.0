import type { NextConfig } from "next";

// FastAPI (kev.serve) is proxied under /kev so the browser never deals with CORS or ports.
const KEV_API = process.env.KEV_API ?? "http://127.0.0.1:8009";

const nextConfig: NextConfig = {
  reactCompiler: true,
  devIndicators: false,
  // Next dev only trusts the hostname it was started with (localhost); without this,
  // opening the app via 127.0.0.1 renders the SSR HTML but never hydrates (no errors, buttons dead).
  allowedDevOrigins: ["127.0.0.1"],
  async rewrites() {
    return [{ source: "/kev/:path*", destination: `${KEV_API}/:path*` }];
  },
};

export default nextConfig;
