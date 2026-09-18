"""Semantic label glossary for Banking77 intent classification.

Maps abbreviated machine-readable category slugs into rich natural-language
semantic descriptions to improve zero-shot and few-shot decision boundaries
in non-autoregressive encoder models.
"""

from __future__ import annotations

BANKING_GLOSSARY: dict[str, str] = {
    # Card Lifecycle & Delivery
    "card_arrival": "Inquire about whether a newly ordered debit or credit card has arrived or when it will arrive in the mail",
    "card_delivery_estimate": "Ask for an estimated delivery date or tracking status for a physical card shipment",
    "card_linking": "Connect or link a new physical or virtual bank card to an account or digital wallet",
    "card_not_working": "Report that a physical card is failing, malfunctioning, chip defective, or unreadable at card terminals",
    "card_about_to_expire": "Inquire about renewal, expiration date, or replacement for a card approaching its expiry",
    "contactless_not_working": "Troubleshoot contactless NFC tap-to-pay functionality failure on a card or terminal",
    "get_physical_card": "Request, order, or purchase a new physical plastic card for an account",
    "pin_blocked": "Unblock or reset a card PIN after multiple incorrect entry attempts at an ATM or terminal",
    "change_pin": "Change, update, or choose a new security PIN for an existing card",
    "compromised_card": "Report suspected unauthorized card cloning, skimmed magnetic stripe, or fraudulent card transactions",
    "lost_or_stolen_card": "Report a physically lost, stolen, or misplaced card requiring immediate permanent block or cancellation",

    # Cash & ATM Operations
    "atm_support": "Find nearby supported ATM machines or ask general questions about cash withdrawals",
    "cash_withdrawal_charge": "Ask about fees, surcharges, or limits applied to cash withdrawals at ATM machines",
    "cash_withdrawal_not_recognised": "Report an unrecognized or unauthorized ATM cash withdrawal transaction on an account statement",
    "declined_cash_withdrawal": "Investigate why an attempted ATM cash withdrawal was rejected, declined, or failed",
    "pending_cash_withdrawal": "Inquire about an ATM cash withdrawal that is stuck in pending status or money was not dispensed",
    "wrong_amount_of_cash_received": "Report an ATM machine dispensing an incorrect physical cash amount compared to the debited total",

    # Transfers & Disbursements
    "transfer_timing": "Ask how long an outgoing or incoming bank transfer, wire, or ACH payment takes to settle",
    "transfer_not_received_by_recipient": "Inquire why sent funds have not yet arrived in the recipient's bank account",
    "failed_transfer": "Investigate why an attempted money transfer failed, bounced, or was rejected by the system",
    "beneficiary_not_allowed": "Troubleshoot errors adding or transferring money to a blocked, restricted, or unauthorized recipient",
    "cancel_transfer": "Request immediate cancellation or clawback of a pending or recently submitted bank transfer",
    "pending_transfer": "Ask why a submitted bank transfer is still marked as pending or processing",
    "transfer_fee_charged": "Inquire about fees or transaction commissions charged on an international or domestic transfer",
    "transfer_into_account": "Ask how to receive money, deposit funds, or wire money into an existing account",

    # Payments & Point of Sale
    "card_payment_fee_charged": "Inquire about unexplained extra fees or service surcharges applied to a card payment",
    "card_payment_not_recognised": "Report an unfamiliar or unrecognized merchant card charge on a bank statement",
    "card_payment_wrong_exchange_rate": "Dispute foreign currency exchange rate calculation applied to a card purchase abroad",
    "declined_card_payment": "Investigate why a merchant card transaction was declined, rejected, or refused at checkout",
    "disposable_virtual_card": "Inquire about creating, using, or resetting single-use disposable virtual cards for online shopping",
    "extra_charge_on_statement": "Dispute an unexpected fee, hidden charge, or unexplained debit appearing on an account statement",
    "pending_card_payment": "Ask why a completed store or online card purchase is still displaying as pending authorization",
    "refund_not_showing_up": "Check status of an expected merchant refund that has not yet credited to the card balance",
    "reverted_card_payment?": "Inquire why a card payment was reversed, cancelled, or money was returned to the balance",

    # Top-ups, Deposits & Balances
    "automatic_top_up": "Set up, manage, or troubleshoot recurring automatic balance replenishment from another card",
    "balance_not_updated_after_cheque_or_cash_deposit": "Inquire why balance has not updated after depositing physical cash or a paper check",
    "balance_not_updated_after_bank_transfer": "Inquire why account balance has not reflected a completed incoming bank wire transfer",
    "check_balance": "Ask how to view, verify, or calculate the current available spending balance in an account",
    "declined_transfer": "Investigate why an incoming or outgoing money transfer was formally declined by the system",
    "edit_personal_details": "Update name, residential address, email, phone number, or KYC profile details",
    "exchange_rate": "Check current foreign exchange rates, conversion spreads, or weekend currency trading markup",
    "fiat_currency_support": "Ask which foreign fiat currencies are supported for holding, exchange, and direct spending",
    "pending_top_up": "Check why a balance top-up or debit card funding transaction is still pending processing",
    "top_up_by_bank_transfer_charge": "Ask about fees or charges associated with topping up an account balance via wire transfer",
    "top_up_by_card_charge": "Ask about processing fees or transaction charges when adding money via debit or credit card",
    "top_up_by_cash_or_cheque": "Inquire about options for depositing physical banknotes or checks into an account balance",
    "top_up_failed": "Troubleshoot why an attempt to add funds or top up an account balance failed or was rejected",
    "top_up_limits": "Ask about daily, weekly, or monthly limits on how much money can be deposited or topped up",
    "top_up_reverted": "Inquire why a deposited top-up was reversed, charged back, or deducted from the balance",
    "unable_to_verify_identity": "Resolve identity verification issues, failed passport/ID photo upload, or KYC document rejection",
    "verify_my_identity": "Ask how to complete identity verification, upload identity documents, or verify residential address",
    "virtual_card_not_working": "Troubleshoot virtual card activation, expiration, or declined online merchant transactions",
    "wrong_exchange_rate_for_cash_withdrawal": "Dispute foreign currency conversion rate applied to an overseas ATM cash withdrawal",
}


def get_enriched_label(category_slug: str) -> str:
    """Return rich natural language description for a category slug.

    If slug has an explicit curated entry, returns it; otherwise formats
    the slug into a natural title-cased sentence.
    """
    if category_slug in BANKING_GLOSSARY:
        return BANKING_GLOSSARY[category_slug]

    # Graceful fallback: format snake_case slug into natural description
    cleaned = category_slug.replace("_", " ").replace("?", "").strip()
    return f"Inquiry or request regarding {cleaned}"
