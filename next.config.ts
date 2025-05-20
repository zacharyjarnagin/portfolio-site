import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  images: {
    remotePatterns: [new URL('https://dmotgjjj6jp2pbpr.public.blob.vercel-storage.com/**')],
  },
};

export default nextConfig;
