// Dev fixtures against the machine-readable VM contract (spec B.3.3, H.3): every static/dev/fixtures/*.json that is
// a series VM, compare VM, C.6 job snapshot or C.7/C.9 envelope validates against trendai/web/vm.schema.json with the
// same JSON-Schema subset as WP1's stdlib validator (testing.validate_schema). Skipped while the schema is absent.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SCHEMA = join(HERE, '../../trendai/web/vm.schema.json');
const FIXTURES = join(HERE, '../../trendai/web/static/dev/fixtures');

const TYPES = {
  object: (v) => v !== null && typeof v === 'object' && !Array.isArray(v),
  array: (v) => Array.isArray(v),
  string: (v) => typeof v === 'string',
  integer: (v) => typeof v === 'number' && Number.isInteger(v),
  number: (v) => typeof v === 'number' && Number.isFinite(v),
  boolean: (v) => typeof v === 'boolean',
  null: (v) => v === null,
};

/**
 * Errors of `instance` against `schema` (empty: valid). Keywords: type (name or array), required, properties,
 * additionalProperties (false), items, enum, minItems, maxItems, $ref ('#/$defs/<name>'). A null instance is valid
 * when the schema's type lists 'null'; no other keyword is checked then.
 */
export function validate(instance, schema, root, path = '$') {
  const errors = [];
  if (schema.type !== undefined) {
    const names = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (instance === null && names.includes('null')) return errors;
    if (!names.some((n) => TYPES[n] && TYPES[n](instance))) {
      return [`${path}: expected ${names.join('|')}, got ${instance === null ? 'null' : Array.isArray(instance) ? 'array' : typeof instance}`];
    }
  }
  if (schema.$ref) {
    const name = schema.$ref.split('/').pop();
    const target = root.$defs && root.$defs[name];
    if (!target) return [`${path}: unknown $ref ${schema.$ref}`];
    errors.push(...validate(instance, target, root, path));
  }
  if (schema.enum && !schema.enum.some((e) => JSON.stringify(e) === JSON.stringify(instance))) {
    errors.push(`${path}: ${JSON.stringify(instance)} not in enum`);
  }
  if (TYPES.object(instance)) {
    const props = schema.properties || {};
    for (const key of schema.required || []) if (!(key in instance)) errors.push(`${path}: missing ${key}`);
    if (schema.additionalProperties === false) {
      for (const key of Object.keys(instance)) if (!(key in props)) errors.push(`${path}: unexpected ${key}`);
    }
    for (const [key, sub] of Object.entries(props)) {
      if (key in instance) errors.push(...validate(instance[key], sub, root, `${path}.${key}`));
    }
  }
  if (Array.isArray(instance)) {
    if (schema.minItems !== undefined && instance.length < schema.minItems) errors.push(`${path}: fewer than ${schema.minItems} items`);
    if (schema.maxItems !== undefined && instance.length > schema.maxItems) errors.push(`${path}: more than ${schema.maxItems} items`);
    if (schema.items) instance.forEach((item, i) => errors.push(...validate(item, schema.items, root, `${path}[${i}]`)));
  }
  return errors;
}

/** The schema entry a fixture belongs to, from its shape; null for responses outside the VM contract. */
export function entryFor(obj) {
  if (!TYPES.object(obj)) return null;
  if ('result_id' in obj && 'series' in obj && 'query' in obj) return 'resultEnvelope';
  if (obj.kind === 'analyze' && 'job_id' in obj) return 'analyzeJob';
  if (obj.kind === 'narrative' && 'job_id' in obj) return 'narrativeJob';
  if ('hero' in obj && 'tiles' in obj) return 'seriesVM';
  if ('ranking' in obj && 'slots' in obj) return 'compareVM';
  return null;
}

const schemaPresent = existsSync(SCHEMA);

test('the validator implements the subset correctly', () => {
  const root = { $defs: { n: { type: ['integer', 'null'] } } };
  const schema = {
    type: 'object', required: ['a'], additionalProperties: false,
    properties: { a: { type: 'array', items: { $ref: '#/$defs/n' }, minItems: 1, maxItems: 2 }, b: { enum: ['x'] } },
  };
  assert.deepEqual(validate({ a: [1, null] }, schema, root), []);
  assert.ok(validate({ a: [1.5] }, schema, root).length);
  assert.ok(validate({ a: [true] }, schema, root).length, 'booleans are not integers');
  assert.ok(validate({ a: [] }, schema, root).length);
  assert.ok(validate({ a: [1], c: 1 }, schema, root).length);
  assert.ok(validate({}, schema, root).length);
  assert.ok(validate({ a: [1], b: 'y' }, schema, root).length);
});

test('every dev fixture validates against vm.schema.json', { skip: schemaPresent ? false : 'trendai/web/vm.schema.json does not exist yet' }, () => {
  const root = JSON.parse(readFileSync(SCHEMA, 'utf8'));
  const files = readdirSync(FIXTURES).filter((f) => f.endsWith('.json')).sort();
  assert.ok(files.length > 0, 'no fixtures');
  let validated = 0;
  const failures = [];
  for (const file of files) {
    const data = JSON.parse(readFileSync(join(FIXTURES, file), 'utf8'));
    const entry = entryFor(data);
    if (!entry) continue;
    const errors = validate(data, { $ref: `#/$defs/${entry}` }, root);
    if (errors.length) failures.push(`${file} (${entry}):\n  ${errors.slice(0, 8).join('\n  ')}`);
    validated += 1;
    // nested VMs inside envelopes / snapshots are covered by the $refs; check the series VMs directly as well
    for (const vm of (entry === 'resultEnvelope' ? data.series : [])) {
      const e = validate(vm, { $ref: '#/$defs/seriesVM' }, root);
      if (e.length) failures.push(`${file} series ${vm.index}: ${e.slice(0, 4).join('; ')}`);
    }
  }
  assert.ok(validated >= 10, `only ${validated} fixtures matched a schema entry`);
  assert.deepEqual(failures, []);
});

test('fixtures are strict JSON without NaN/Infinity or Python reprs in texts', () => {
  for (const file of readdirSync(FIXTURES).filter((f) => f.endsWith('.json'))) {
    const raw = readFileSync(join(FIXTURES, file), 'utf8');
    assert.ok(!/\bNaN\b|\bInfinity\b/.test(raw), `${file} has NaN/Infinity`);
    assert.ok(!/"[^"]*\b(None|nan|undefined)\b[^"]*"/.test(raw.replace(/"(?:[^"\\]|\\.)*":/g, '')), `${file} has None/nan/undefined in a string`);
  }
});
