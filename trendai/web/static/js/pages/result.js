// Result pages (spec F.6.11): /r/<id> polls the job while it runs (progress as F.6.5) or loads the stored result,
// adds it to recent analyses (with expires_at) and, for upload results, shows the expiry line with a 공유 button;
// /s/<id> loads GET /api/shared/<id> with the read-only banner (no share or AI buttons; exports stay available).
// The rendering is shared with /explore (pages/explore.js). Owner: WP3-app.

import { createResultsPage } from './explore.js';

export function create(ctx) {
  return createResultsPage(ctx, ctx.state.route === 'shared' ? 'shared' : 'result');
}
