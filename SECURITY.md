# Security policy

Report a vulnerability privately through GitHub's "Report a
vulnerability" button on this repository's Security tab. Please do not
open a public issue for it. You will get a reply within 7 days.

Scope: the code in this repository. It is an offline measurement
harness: it trains models on a public dataset and reads and writes
local JSON files. It runs no service and stores no credential. Model
weights are fetched by torchvision from PyTorch's official host.
