import "./globals.css";

export const metadata = {
  title: "Three Eyed Raven | Crowd Intelligence",
  description: "Real-time crowd tracking project dashboard",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
