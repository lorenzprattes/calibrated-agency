# Security

Contact the repository owner privately to report a vulnerability. Do not put API keys,
private prompts, or working exploit details in a public issue.

Model adapters send prompts to the endpoint configured by the caller. The application is
responsible for endpoint access, transport security, and data retention.

The package rejects missing or nonfinite scores, conflicting candidate scores, duplicate
token IDs, and unsupported labels. The API adapter also checks that generation used the
same prompt tokens as tokenization.

Keep credentials out of cache identities, logs, and test fixtures. Supply them through
an environment variable or an external credential provider.
