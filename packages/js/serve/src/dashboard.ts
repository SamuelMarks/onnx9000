import { Router } from './router';

// 145. Provide a built-in interactive HTML dashboard available at `/v2/dashboard`.
export function addDashboardRoutes(router: Router) {
  router.get('/v2/dashboard', async () => {
    const html = `<!DOCTYPE html>
<html>
<head>
  <title>ONNX9000 Dashboard</title>
  <!-- 154. Provide strict Content Security Policy (CSP) headers -->
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline' https://d3js.org; style-src 'self' 'unsafe-inline';">
  <style>
    body { font-family: sans-serif; background: #121212; color: #eee; margin: 0; padding: 20px; }
    h1 { color: #fff; }
    .card { background: #1e1e1e; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
  </style>
</head>
<body>
  <h1>ONNX9000 Edge Server</h1>
  <div class="card">
    <h2>Metrics Overview</h2>
    <div id="metrics-content">Loading...</div>
  </div>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script>
    fetch('/metrics').then(r => r.text()).then(txt => {
      document.getElementById('metrics-content').innerText = txt;
    });
  </script>
</body>
</html>`;

    return new Response(html, {
      status: 200,
      headers: {
        'Content-Type': 'text/html; charset=utf-8',
        'Content-Security-Policy':
          "default-src 'self'; script-src 'self' 'unsafe-inline' https://d3js.org; style-src 'self' 'unsafe-inline';",
      },
    });
  });
}
