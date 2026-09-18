# Decision Engine Failure Gallery and Boundary Audit

Documenting empirical model errors, false acceptances, and boundary edge cases across held-out test data and challenge slices.

| Slice / Split | Case ID | Customer Utterance | Expected | Predicted | Confidence | Above 85% Gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `held_out_test` | `test_in_00453` | Is a copy of the police report necessary for com... | `lost_or_stolen_card` | `__insufficient_evidence__` | 99.1% | Yes (Bypasses Gate) |
| `held_out_test` | `test_in_00037` | Is there tracking info available?... | `card_arrival` | `__insufficient_evidence__` | 96.7% | Yes (Bypasses Gate) |
| `held_out_test` | `test_in_00623` | How much is the fee for a SEPA transfer?... | `top_up_by_bank_transfer_charge` | `transfer_fee_charged` | 93.8% | Yes (Bypasses Gate) |
| `held_out_test` | `test_in_00799` | Will declined funds I tried to withdraw be retur... | `wrong_amount_of_cash_received` | `declined_cash_withdrawal` | 93.7% | Yes (Bypasses Gate) |
| `held_out_test` | `test_in_00333` | where is theft-top option?... | `automatic_top_up` | `__insufficient_evidence__` | 92.0% | Yes (Bypasses Gate) |
| `held_out_test` | `test_in_00773` | Where is the money I pushed it's on my mobile ap... | `wrong_amount_of_cash_received` | `__insufficient_evidence__` | 90.9% | Yes (Bypasses Gate) |
| `held_out_test` | `test_in_00610` | Do I have to pay any fees in order to receive mo... | `top_up_by_bank_transfer_charge` | `receiving_money` | 90.5% | Yes (Bypasses Gate) |
| `held_out_test` | `test_oos_00175` | what's the extended zipcode for my address... | `__insufficient_evidence__` | `edit_personal_details` | 86.9% | Yes (Bypasses Gate) |
| `held_out_test` | `test_in_00232` | I tried to take money from my card, but it didn'... | `pending_cash_withdrawal` | `card_swallowed` | 83.9% | No (Safely Gated) |
| `held_out_test` | `test_oos_00165` | i need to add a person to my bank account... | `__insufficient_evidence__` | `get_physical_card` | 83.9% | No (Safely Gated) |
| `held_out_test` | `test_in_00658` | Is there something wrong with your website? I tr... | `pending_top_up` | `topping_up_by_card` | 83.1% | No (Safely Gated) |
| `held_out_test` | `test_in_00683` | I made a transaction but did it to the wrong acc... | `cancel_transfer` | `beneficiary_not_allowed` | 81.1% | No (Safely Gated) |
| `held_out_test` | `test_in_00072` | Can I reactivate a card I thought I lost?... | `card_linking` | `card_swallowed` | 74.4% | No (Safely Gated) |
| `held_out_test` | `test_in_00268` | I want to make a currency exchange to EU.... | `fiat_currency_support` | `__insufficient_evidence__` | 74.3% | No (Safely Gated) |
| `held_out_test` | `test_oos_00139` | how do i add someone to my account... | `__insufficient_evidence__` | `get_physical_card` | 49.3% | No (Safely Gated) |
| `held_out_test` | `test_in_00704` | I made a mistake with a transaction!... | `cancel_transfer` | `__insufficient_evidence__` | 49.3% | No (Safely Gated) |
| `held_out_test` | `test_in_00630` | Please tell me about SWIFT transfers at this ban... | `top_up_by_bank_transfer_charge` | `__insufficient_evidence__` | 49.2% | No (Safely Gated) |
| `held_out_test` | `test_in_00276` | How do I accept exchanges to EU?... | `fiat_currency_support` | `exchange_charge` | 48.2% | No (Safely Gated) |
| `missing_option` | `test_mo_00006` | Why was I charged for card payment?... | `__insufficient_evidence__` | `extra_charge_on_statement` | 91.9% | Yes (Bypasses Gate) |
| `missing_option` | `test_mo_00013` | What is the fee charged with this card payment?... | `__insufficient_evidence__` | `transfer_fee_charged` | 87.6% | Yes (Bypasses Gate) |
| `missing_option` | `test_mo_00025` | How do I avoid getting charged a fee on my card?... | `__insufficient_evidence__` | `card_payment_not_recognised` | 72.5% | No (Safely Gated) |
| `missing_option` | `test_mo_00034` | Why was I charged a fee when I paid with card?... | `__insufficient_evidence__` | `transfer_fee_charged` | 98.4% | Yes (Bypasses Gate) |
| `missing_option` | `test_mo_00038` | Why was a charged a fee for using the card?... | `__insufficient_evidence__` | `extra_charge_on_statement` | 86.0% | Yes (Bypasses Gate) |
| `missing_option` | `test_mo_00039` | They charged me for paying with my card.... | `__insufficient_evidence__` | `extra_charge_on_statement` | 82.4% | No (Safely Gated) |
| `missing_option` | `test_mo_00040` | I completed an in country transfer a few days ag... | `__insufficient_evidence__` | `balance_not_updated_after_bank_transfer` | 96.6% | Yes (Bypasses Gate) |
| `missing_option` | `test_mo_00044` | I tried to send someone money but they haven't r... | `__insufficient_evidence__` | `balance_not_updated_after_bank_transfer` | 91.3% | Yes (Bypasses Gate) |

### Empirical Gating Analysis

Of the 50 total test set classification errors (50/1000 = 5.00% error rate):

1. Errors above 85% confidence: 10 (20.0% of errors) exhibited confidence >= 85% and would bypass an autonomous policy threshold of 85%.
2. Errors below 85% confidence: 40 (80.0% of errors) were produced with confidence < 85% and would be safely routed to human supervisor review.

An autonomous threshold at 85% intercepts the majority of lower-confidence errors, but near-sibling confusions frequently exhibit overconfident incorrect predictions.
