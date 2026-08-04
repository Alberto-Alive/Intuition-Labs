## Failure analysis

### Failure counts by case type (final-answer errors)

| Case type | Grep | Vector RAG | Hybrid RAG | GrepRAG | DIGIT | DIGIT (no gate) |
|---|---|---|---|---|---|---|
| alias | 0/18 | 2/18 | 2/18 | 11/18 | 0/18 | 15/18 |
| ambiguous_keyword | 0/18 | 0/18 | 0/18 | 1/18 | 0/18 | 0/18 |
| approved_rejected | 18/18 | 4/18 | 6/18 | 17/18 | 0/18 | 0/18 |
| code_symbol | 6/18 | 1/18 | 0/18 | 10/18 | 0/18 | 0/18 |
| conflict | 0/18 | 2/18 | 3/18 | 5/18 | 0/18 | 9/18 |
| date | 0/18 | 0/18 | 0/18 | 0/18 | 0/18 | 0/18 |
| exception | 3/18 | 18/18 | 15/18 | 18/18 | 0/18 | 18/18 |
| multi_doc | 13/18 | 4/18 | 17/18 | 12/18 | 0/18 | 0/18 |
| policy_condition | 6/18 | 6/18 | 3/18 | 6/18 | 0/18 | 0/18 |
| stale_current | 18/18 | 0/18 | 0/18 | 15/18 | 0/18 | 1/18 |
| table | 15/18 | 10/18 | 5/18 | 16/18 | 0/18 | 0/18 |
| unanswerable | 2/18 | 2/18 | 3/18 | 12/18 | 0/18 | 0/18 |

### Representative traps (grep finds the literal, but it does not apply)

**[stale_current] What is the current refund window for Lyra-001?**
- Gold: `VALUE` / value `30 days`  
- Distractor literal grep is tempted by: `14 days`  
- **Grep** → `YES` value `14 days`: The current refund window for Lyra-001 is 14 days as per Policy Lyra-001-RP.  
- **DIGIT** → `VALUE` value `30 days`: The current refund window for Lyra-001 is 30 days. This information can be found in document D0002.

**[approved_rejected] Is the 90 day data retention from proposal P-2002 now in effect for Ember-002?**
- Gold: `NO` / value `90`  
- Distractor literal grep is tempted by: `90`  
- **Grep** → `YES` value `—`: Yes, the 90-day data retention policy specified in Proposal P-2002 has been implemented for Ember-002.  
- **DIGIT** → `NO` value `not in effect (proposal rejected)`: The policy does not specify that the 90-day data retention period applies to Ember-002.

**[stale_current] What is the current refund window for Marlin-013?**
- Gold: `VALUE` / value `30 days`  
- Distractor literal grep is tempted by: `45 days`  
- **Grep** → `INSUFFICIENT` value `45 days`: The current refund window for Marlin-013 is 45 days.  
- **DIGIT** → `VALUE` value `30 days`: The current refund window for Marlin-013 is 30 days. This information can be found in document D0022.

**[approved_rejected] Is the 5 TB storage cap from proposal P-2014 now in effect for Vertex-014?**
- Gold: `NO` / value `5`  
- Distractor literal grep is tempted by: `5`  
- **Grep** → `YES` value `—`: Yes, the 5 TB storage cap from proposal P-2014 has been implemented as of the current state of the record.  
- **DIGIT** → `NO` value `not in effect (proposal rejected)`: The policy field indicates that proposal P-2014 was not in effect (rejected), so it cannot be applied to Vertex-014.

**[multi_doc] Where are backups physically stored for a Onyx-018 account?**
- Gold: `VALUE` / value `eu-west-1`  
- Distractor literal grep is tempted by: `us-east-1`  
- **Grep** → `YES` value `sa-east-1`: Backups for a Onyx-018 account are physically stored in sa-east-1.  
- **DIGIT** → `YES` value `eu-west-1`: Backups for a Onyx-018 account are physically stored in the region specified as "eu-west-1".

**[table] In the Harbor-019 pricing matrix, what is the monthly price of the government plan in APAC?**
- Gold: `VALUE` / value `$499`  
- Distractor literal grep is tempted by: `$499`  
- **Grep** → `INSUFFICIENT` value `—`: The context does not provide a direct monthly price for the government plan in APAC. Therefore, the answer is:  
- **DIGIT** → `YES` value `$499`: The monthly price of the government plan in APAC for Harbor-019 is $499.

**[stale_current] What is the current refund window for Cobalt-025?**
- Gold: `VALUE` / value `30 days`  
- Distractor literal grep is tempted by: `45 days`  
- **Grep** → `YES` value `45 days`: The current refund window for Cobalt-025 is 45 days as per policy Cobalt-025-RP.  
- **DIGIT** → `VALUE` value `30 days`: The current refund window for Cobalt-025 is 30 days. This information can be found in document D0042.

**[approved_rejected] Is the 60 day refund window from proposal P-2026 now in effect for Atlas-026?**
- Gold: `NO` / value `60`  
- Distractor literal grep is tempted by: `60`  
- **Grep** → `YES` value `—`: Yes, the 60-day refund window proposed in Proposal P-2026 has been implemented for Atlas-026 as indicated by the policy changes in D0043 and D0163.  
- **DIGIT** → `NO` value `not in effect (proposal withdrawn)`: The policy field indicates that the proposal P-2026 is not in effect (withdrawn), so it cannot be related to the current situation involving the Atlas-026 propo
