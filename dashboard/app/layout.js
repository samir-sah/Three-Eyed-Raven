import "./globals.css";
import "./workspace.css";

export const metadata = {
  title: "Three Eyed Raven | Crowd Intelligence",
  description: "Analyse your uploaded aerial and ground videos with anonymous crowd detection",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
