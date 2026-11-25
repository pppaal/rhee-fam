// Chart descriptions for <desc> and screen readers (spec F.7.5). Owner: WP3-chart.
// Pure: no DOM access. Used by fanchart.js, comparechart.js and views (WP3-app) for aria text.
// Catalog keys used through `t` (WP3-app adds them to catalog.js, see docs/WEB_INTERFACES.md):
//   'chart.describe.compare'  ko '{n}개 시계열 비교: {names}.'  en 'Comparison of {n} series: {names}.'
//   'chart.describe.top'      ko '상승 점수 1위 {name}.'          en 'Highest rising score: {name}.'

/**
 * Description of a single-series chart: the VM's `chart.desc`, unchanged.
 * @param {Object} seriesVM  D.2 series VM
 * @returns {string}
 */
export function describeSeries(seriesVM) {
  return (seriesVM && seriesVM.chart && seriesVM.chart.desc) || '';
}

/**
 * Description of a comparison chart: t('chart.describe.compare', {n, names}) + ' ' + t('chart.describe.top',
 * {name}) where `names` are the series names in request order joined with ', ' and `name` is the name of
 * compareVM.ranking[0] (the top rising score). Without ranking rows only the first sentence.
 * @param {Object} compareVM   D.4 compare VM
 * @param {Object[]} seriesVMs D.2 series VMs of the successful series, request order
 * @param {(key: string, params?: Object) => string} t
 * @returns {string}
 */
export function describeCompare(compareVM, seriesVMs, t) {
  const series = (seriesVMs || []).slice().sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
  const names = series.map((vm) => vm.name).filter(Boolean);
  const first = t('chart.describe.compare', { n: names.length, names: names.join(', ') });
  const top = compareVM && Array.isArray(compareVM.ranking) && compareVM.ranking.length ? compareVM.ranking[0] : null;
  return top && top.name ? `${first} ${t('chart.describe.top', { name: top.name })}` : first;
}
