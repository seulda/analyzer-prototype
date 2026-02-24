import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Roof Analyzer",
  description: "지붕 구조 분석 프로토타입",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
