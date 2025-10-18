const localtunnel = require('localtunnel');

(async () => {
  try {
    // change subdomain if taken, or remove `subdomain` to get a random URL
    const tunnel = await localtunnel({ port: 8000, subdomain: 'humanhelperai' });

    console.log('Public URL:', tunnel.url);
    console.log('Press Ctrl+C to exit. Tunnel will remain until this process exits.');

    tunnel.on('close', () => {
      console.log('Tunnel closed.');
      process.exit(0);
    });

    // Optional: print incoming requests (uncomment to log)
    // tunnel.on('request', (req) => console.log('req', req));
  } catch (err) {
    console.error('localtunnel failed:', err);
    process.exit(1);
  }
})();
