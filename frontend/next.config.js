/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy API calls to FastAPI backend to avoid CORS in development
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
