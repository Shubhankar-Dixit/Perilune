import "../styles/globals.css";
import type { Metadata } from "next";
import TopNav from "../components/TopNav";
import { Inter, Playfair_Display } from "next/font/google";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  weight: ["300", "400", "500", "600", "700"],
  display: "swap",
});

const display = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["400", "700"],
  style: ["normal", "italic"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Perilune Explorer",
  description: "Interact with exoplanet transit models and insights",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${display.variable}`}>
      <body>
        <TopNav />
        <div className="container" style={{ paddingTop: "1.25rem", paddingBottom: "2rem" }}>
          {children}
        </div>
      </body>
    </html>
  );
}

