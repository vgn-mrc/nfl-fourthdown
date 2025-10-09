export const metadata = {
  title: 'NFL 4th Down Demo',
  description: 'GO/FG/PUNT model inference demo',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial' }}>
        <div style={{ maxWidth: 960, margin: '24px auto', padding: '0 16px' }}>
          <h1 style={{ marginBottom: 8 }}>NFL 4th Down Inference</h1>
          <p style={{ color: '#555', marginTop: 0 }}>Enter situation → get WP, components, and coach policy.</p>
          {children}
        </div>
      </body>
    </html>
  );
}

