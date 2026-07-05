import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Thumbnail Scorer",
  description: "Score your YouTube thumbnails with AI",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col" style={{ backgroundColor: 'var(--bg)', color: 'var(--text)' }}>
        {children}
      </body>
    </html>
  );
}
