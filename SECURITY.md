# Security policy

## Reporting

Until a public maintainer address is designated, report vulnerabilities privately to the
repository owner rather than opening a public issue containing exploit details or secrets.

## Scope

Backend adapters send prompts and model outputs to configured inference services. Operators
are responsible for endpoint access controls, transport security, retention, and data-use
policies. Do not send sensitive prompts to an endpoint whose operator or logging policy is
unknown.

The package treats model and endpoint responses as untrusted. Missing scores, malformed
responses, duplicate token IDs, unsupported token boundaries, and unknown score modes must
fail closed.

Never include API keys in cache identities, logs, fixtures, or bug reports. Use environment
variables or an external credential provider.
