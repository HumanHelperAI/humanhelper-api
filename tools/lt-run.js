#!/usr/bin/env node
// tools/lt-run.js
// Programmatic LocalTunnel runner that prints public URL + tunnel password
// Usage: node tools/lt-run.js --port 8000 --subdomain humanhelperai

const localtunnel = require('localtunnel');
const fetch = require('node-fetch'); // v2 style

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { port: 8000, subdomain: undefined };
  for (let i = 0; i < args.length; i++) {
    if ((args[i] === '--port' || args[i] === '-p') && args[i+1]) { opts.port = Number(args[++i]); }
    else if ((args[i] === '--subdomain' || args[i] === '-s') && args[i+1]) { opts.subdomain = args[++i]; }
    else if (args[i] === '--help' || args[i] === '-h') { console.log('Usage: node tools/lt-run.js --port 8000 --subdomain name'); process.exit(0); }
  }
  return opts;
}

(async () => {
  try {
    const opts = parseArgs();
    console.log(`[lt-run] starting localtunnel on port ${opts.port} ...`);

    const tunnel = await localtunnel({ port: opts.port, subdomain: opts.subdomain });
    // tunnel.url e.g. https://humanhelperai.loca.lt
    console.log(`[lt-run] public URL: ${tunnel.url}`);

    // Save info to files so shell helpers can read them
    const fs = require('fs');
    const baseDir = process.env.HOME + '/.localtunnel_humanhelper';
    try { fs.mkdirSync(baseDir, { recursive: true }); } catch (e) {}
    fs.writeFileSync(baseDir + '/url', tunnel.url);
    fs.writeFileSync(baseDir + '/pid', String(process.pid));

    // Wait a little then attempt to fetch the tunnel password (loca.lt provides /mytunnelpassword)
    // This endpoint returns the tunnel password only if request originates from same host or service allows it.
    // We try to fetch and print it (best-effort).
    const pwEndpoint = 'https://loca.lt/mytunnelpassword';
    try {
      // small delay
      await new Promise(r => setTimeout(r, 900));
      const resp = await fetch(pwEndpoint, { timeout: 5000 });
      if (resp.ok) {
        const text = (await resp.text()).trim();
        if (text) {
          console.log(`[lt-run] tunnel password: ${text}`);
          fs.writeFileSync(baseDir + '/password', text);
        } else {
          console.log('[lt-run] tunnel password: (empty response)');
        }
      } else {
        console.log(`[lt-run] cannot fetch password: HTTP ${resp.status}`);
      }
    } catch (err) {
      console.log('[lt-run] fetch password failed (this is OK):', err.message || err);
    }

    console.log('[lt-run] tunnel running — press Ctrl+C to end or use the stop helper.');

    // forward close to exit gracefully
    tunnel.on('close', () => {
      try { fs.unlinkSync(baseDir + '/url'); } catch(_) {}
      try { fs.unlinkSync(baseDir + '/pid'); } catch(_) {}
      try { fs.unlinkSync(baseDir + '/password'); } catch(_) {}
      console.log('[lt-run] tunnel closed');
      process.exit(0);
    });

    // keep process alive
  } catch (e) {
    console.error('[lt-run] fatal error:', e && e.stack ? e.stack : String(e));
    process.exit(1);
  }
})();
