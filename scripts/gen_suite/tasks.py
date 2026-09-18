"""Task registry and schema definitions for the OpenJev 12-task cookbook sweep.

Defines the single shared TaskItem shape and candidate descriptions across
three label-framing arms: neutral, verbose, and banking_framed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

FramingArm = Literal["neutral", "verbose", "banking_framed"]
ProvenanceType = Literal["public", "authored"]

INSUFFICIENT_EVIDENCE_ID = "__insufficient_evidence__"


@dataclass
class CandidateOption:
    id: str
    description: str


@dataclass
class TaskItem:
    task: str
    id: str
    question: str
    context: str
    candidates: list[dict[str, str]]
    target_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskItem:
        return cls(
            task=data["task"],
            id=data["id"],
            question=data["question"],
            context=data["context"],
            candidates=data["candidates"],
            target_id=data["target_id"],
            meta=data.get("meta", {}),
        )


@dataclass
class TaskSpec:
    id: str
    name: str
    cookbook: str
    source: str
    k_cardinality: int
    demo_value: str
    description: str
    provenance: ProvenanceType
    provenance_details: str
    default_question: str
    # Candidates dictionary: {arm: {candidate_id: description}}
    candidate_arms: dict[FramingArm, dict[str, str]]


# ---------------------------------------------------------------------------
# Task 01: Phishing vs legitimate email
# ---------------------------------------------------------------------------
T01_ARMS: dict[FramingArm, dict[str, str]] = {
    "neutral": {
        "phishing": "Phishing or fraudulent email",
        "legitimate": "Legitimate email correspondence",
        INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence to determine email type",
    },
    "verbose": {
        "phishing": "Fraudulent phishing or credential harvesting email designed to deceive recipient via spoofed links, urgent threats, or malicious payloads",
        "legitimate": "Legitimate, safe interpersonal communication, business correspondence, newsletter, or verified account notification",
        INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence or ambiguous email content preventing reliable security determination",
    },
    "banking_framed": {
        "phishing": "Fraudulent financial or banking phishing email attempting unauthorized account takeover, wire fraud, or banking credential theft",
        "legitimate": "Legitimate bank account alert, customer statement, transaction receipt, or verified personal message",
        INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence to classify as financial phishing or legitimate email",
    },
}

# ---------------------------------------------------------------------------
# Task 02: Jailbreak vs benign prompt
# ---------------------------------------------------------------------------
T02_ARMS: dict[FramingArm, dict[str, str]] = {
    "neutral": {
        "jailbreak": "Adversarial jailbreak prompt",
        "benign": "Benign instructional user prompt",
        INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence to assess prompt intent",
    },
    "verbose": {
        "jailbreak": "Adversarial prompt injection, jailbreak attack, or persona override designed to bypass safety guardrails and content boundaries",
        "benign": "Standard, safe user instruction, factual question, coding request, or creative writing prompt",
        INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence to determine whether prompt contains adversarial exploit intent",
    },
    "banking_framed": {
        "jailbreak": "Adversarial prompt injection attack attempting to override banking policy safeguards, access restricted customer data, or bypass financial controls",
        "benign": "Standard customer banking inquiry, account assistance request, or routine financial task instruction",
        INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence to classify as security bypass exploit or benign banking query",
    },
}

# ---------------------------------------------------------------------------
# Task 03: Hazard severity routing
# ---------------------------------------------------------------------------
T03_ARMS: dict[FramingArm, dict[str, str]] = {
    "neutral": {
        "benign": "Benign safe request",
        "low_risk": "Low-risk prompt with minor policy ambiguity",
        "high_risk_jailbreak": "High-risk adversarial jailbreak or prompt injection",
        "critical_security_threat": "Critical severe security threat or harmful exploit",
        INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence to assess hazard severity",
    },
    "verbose": {
        "benign": "Completely benign user request with zero safety or security hazards, safe to process automatically",
        "low_risk": "Low-risk borderline query with minor policy sensitivity, safe with standard output logging",
        "high_risk_jailbreak": "Adversarial prompt injection or sophisticated jailbreak attempting policy bypass, requiring defensive blocking",
        "critical_security_threat": "Critical malicious threat targeting system compromise, malware generation, or illegal harm execution",
        INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence or incomplete context preventing definitive hazard severity scoring",
    },
    "banking_framed": {
        "benign": "Standard benign banking transaction or balance query with no financial security hazard",
        "low_risk": "Low-risk ambiguous financial query requiring standard audit logging",
        "high_risk_jailbreak": "Adversarial security attack attempting to breach financial guardrails or extract confidential banking rules",
        "critical_security_threat": "Critical fraudulent exploit attempting unauthorized funds transfer, credential theft, or system takeover",
        INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence to route financial safety hazard level",
    },
}

# ---------------------------------------------------------------------------
# Task 04: Shopify product taxonomy, flat leaf (24 categories + abstention = 25)
# ---------------------------------------------------------------------------
SHOPIFY_CATEGORIES_NEUTRAL = {
    "apparel_accessories": "Apparel & Accessories",
    "electronics": "Electronics & Gadgets",
    "home_garden": "Home & Garden Furnishings",
    "sporting_goods": "Sporting Goods & Fitness",
    "health_beauty": "Health & Beauty Personal Care",
    "toys_games": "Toys & Games",
    "food_beverages": "Food & Beverages",
    "baby_toddler": "Baby & Toddler Supplies",
    "business_industrial": "Business & Industrial Equipment",
    "hardware_tools": "Hardware & Tools",
    "automotive_parts": "Automotive Parts & Accessories",
    "office_supplies": "Office Supplies & Stationery",
    "pet_supplies": "Pet Supplies & Animal Care",
    "arts_entertainment": "Arts & Crafts Hobbies",
    "luggage_bags": "Luggage, Backpacks & Bags",
    "cameras_optics": "Cameras & Optics",
    "media_books": "Books, Music & Software",
    "jewelry_watches": "Jewelry & Watches",
    "furniture": "Home & Office Furniture",
    "shoes_footwear": "Shoes & Footwear",
    "kitchen_dining": "Kitchen & Dining Ware",
    "lighting": "Lamps & Lighting Fixtures",
    "musical_instruments": "Musical Instruments & Audio Gear",
    "safety_security": "Safety & Security Equipment",
    INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence to match listed retail categories",
}

SHOPIFY_CATEGORIES_VERBOSE = {
    k: f"Retail e-commerce category for {v}" if k != INSUFFICIENT_EVIDENCE_ID else "Insufficient product description or ambiguous item preventing confident retail classification"
    for k, v in SHOPIFY_CATEGORIES_NEUTRAL.items()
}

SHOPIFY_CATEGORIES_BANKING = {
    k: f"Commercial merchant spending category for {v}" if k != INSUFFICIENT_EVIDENCE_ID else "Insufficient transaction description to categorize commercial retail purchase"
    for k, v in SHOPIFY_CATEGORIES_NEUTRAL.items()
}

T04_ARMS: dict[FramingArm, dict[str, str]] = {
    "neutral": SHOPIFY_CATEGORIES_NEUTRAL,
    "verbose": SHOPIFY_CATEGORIES_VERBOSE,
    "banking_framed": SHOPIFY_CATEGORIES_BANKING,
}

# ---------------------------------------------------------------------------
# Task 05: Shopify taxonomy, hierarchical beam
# ---------------------------------------------------------------------------
T05_TOP_NEUTRAL = {
    "apparel_root": "Apparel, Clothing, Shoes and Accessories",
    "home_goods_root": "Home, Garden, Kitchen and Furniture",
    "tech_electronics_root": "Electronics, Computers, Cameras and Audio",
    "leisure_sports_root": "Sports, Fitness, Outdoor and Toys",
    "health_food_root": "Health, Beauty, Food and Baby Supplies",
    "industrial_auto_root": "Business, Industrial, Automotive and Tools",
    INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence for top-level retail division",
}

T05_ARMS: dict[FramingArm, dict[str, str]] = {
    "neutral": T05_TOP_NEUTRAL,
    "verbose": {
        k: f"Broad retail commerce division grouping {v}" if k != INSUFFICIENT_EVIDENCE_ID else "Insufficient evidence to classify merchandise into major retail department"
        for k, v in T05_TOP_NEUTRAL.items()
    },
    "banking_framed": {
        k: f"Merchant category code (MCC) retail department for {v}" if k != INSUFFICIENT_EVIDENCE_ID else "Insufficient evidence for merchant category code assignment"
        for k, v in T05_TOP_NEUTRAL.items()
    },
}

# ---------------------------------------------------------------------------
# Task 06: Banking77 in-domain control (5 options per item dynamically loaded)
# ---------------------------------------------------------------------------
T06_ARMS: dict[FramingArm, dict[str, str]] = {
    "neutral": {
        "card_arrival": "Inquire about whether a newly ordered debit or credit card has arrived or when it will arrive in the mail",
        "card_linking": "Connect or link a new physical or virtual bank card to an account or digital wallet",
        "card_payment_fee_charged": "Inquire about unexplained extra fees or service surcharges applied to a card payment",
        "compromised_card": "Report suspected unauthorized card cloning, skimmed magnetic stripe, or fraudulent card transactions",
        INSUFFICIENT_EVIDENCE_ID: "insufficient evidence",
    },
    "verbose": {
        "card_arrival": "Inquire about whether a newly ordered debit or credit card has arrived or when it will arrive in the mail",
        "card_linking": "Connect or link a new physical or virtual bank card to an account or digital wallet",
        "card_payment_fee_charged": "Inquire about unexplained extra fees or service surcharges applied to a card payment",
        "compromised_card": "Report suspected unauthorized card cloning, skimmed magnetic stripe, or fraudulent card transactions",
        INSUFFICIENT_EVIDENCE_ID: "insufficient evidence",
    },
    "banking_framed": {
        "card_arrival": "Inquire about whether a newly ordered debit or credit card has arrived or when it will arrive in the mail",
        "card_linking": "Connect or link a new physical or virtual bank card to an account or digital wallet",
        "card_payment_fee_charged": "Inquire about unexplained extra fees or service surcharges applied to a card payment",
        "compromised_card": "Report suspected unauthorized card cloning, skimmed magnetic stripe, or fraudulent card transactions",
        INSUFFICIENT_EVIDENCE_ID: "insufficient evidence",
    },
}

# ---------------------------------------------------------------------------
# Task 07: CLINC150 intent routing across 10 functional domains
# ---------------------------------------------------------------------------
T07_DOMAINS_NEUTRAL = {
    "banking": "Banking, deposits, transfers, and account balance management",
    "credit_cards": "Credit card management, limits, rewards, and interest",
    "dining": "Restaurant reservations, meal planning, and nutrition",
    "travel": "Flight bookings, hotel reservations, and travel itineraries",
    "home_utility": "Smart home controls, appliance utilities, and bills",
    "auto_commute": "Vehicle maintenance, traffic updates, and commute navigation",
    "work_productivity": "Calendar scheduling, email management, and meeting agendas",
    "small_talk": "Casual social conversation and greetings",
    "meta_assistant": "Assistant capabilities, settings, and command clarification",
    "general_knowledge": "Factual knowledge, weather forecasts, and dictionary definitions",
    INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence to route into supported assistant domains",
}

T07_ARMS: dict[FramingArm, dict[str, str]] = {
    "neutral": T07_DOMAINS_NEUTRAL,
    "verbose": {
        k: f"AI virtual assistant domain handling {v}" if k != INSUFFICIENT_EVIDENCE_ID else "Insufficient evidence or ambiguous query preventing assistant domain routing"
        for k, v in T07_DOMAINS_NEUTRAL.items()
    },
    "banking_framed": {
        k: f"Multi-service enterprise banking portal routing request to {v}" if k != INSUFFICIENT_EVIDENCE_ID else "Insufficient evidence to route enterprise banking portal intent"
        for k, v in T07_DOMAINS_NEUTRAL.items()
    },
}

# ---------------------------------------------------------------------------
# Task 08: Smart home command interpretation (8 actions, authored fixture)
# ---------------------------------------------------------------------------
T08_ACTIONS_NEUTRAL = {
    "lights_control": "Turn on, turn off, or dim household lights",
    "climate_control": "Adjust thermostat temperature, AC, or heating",
    "security_locks": "Lock or unlock doors and arm security system",
    "media_playback": "Play music, pause podcast, or adjust speaker volume",
    "vacuum_cleaning": "Start robot vacuum cleaner in specific rooms",
    "alarms_timers": "Set alarm, wake-up timer, or reminder countdown",
    "camera_feed": "Show security camera video feed or doorbell camera",
    INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence to trigger a home automation command",
}

T08_ARMS: dict[FramingArm, dict[str, str]] = {
    "neutral": T08_ACTIONS_NEUTRAL,
    "verbose": {
        k: f"Home automation controller action to {v}" if k != INSUFFICIENT_EVIDENCE_ID else "Insufficient evidence or conversational statement with no actionable smart home trigger"
        for k, v in T08_ACTIONS_NEUTRAL.items()
    },
    "banking_framed": {
        k: f"Secure residential facility control action to {v}" if k != INSUFFICIENT_EVIDENCE_ID else "Insufficient evidence to authorize residential facility automation action"
        for k, v in T08_ACTIONS_NEUTRAL.items()
    },
}

# ---------------------------------------------------------------------------
# Task 09: Function and tool routing (20 typed functions + abstention = 21, authored fixture)
# ---------------------------------------------------------------------------
T09_FUNCTIONS_NEUTRAL = {
    "plot_price": "Chart stock historical price line or candlesticks",
    "market_summary": "Get broad market index summary and performance",
    "compare_returns": "Compare percentage returns across multiple tickers",
    "rolling_correlation": "Calculate rolling correlation between symbol and benchmark",
    "volatility_calc": "Compute historical price volatility and annualized standard deviation",
    "summary_stats": "Fetch fundamental financial summary statistics",
    "top_movers": "List top daily market gainers or losers",
    "drawdown_analysis": "Calculate maximum drawdown and recovery period",
    "intraday_pattern": "Analyze intraday hourly volume and price pattern",
    "list_symbols": "List available tradeable ticker symbols in the exchange",
    "execute_trade": "Submit market or limit buy/sell order for execution",
    "cancel_order": "Cancel an active pending unfilled limit order",
    "portfolio_balance": "Check cash balance and total portfolio equity value",
    "position_details": "View current open position quantity and unrealized PnL",
    "transfer_funds": "Transfer funds between connected checking and brokerage accounts",
    "dividend_history": "Look up past historical dividend payout dates and yields",
    "option_chain": "Retrieve call and put options strike prices and greeks",
    "crypto_quote": "Fetch real-time cryptocurrency exchange spot prices",
    "forex_rate": "Query foreign currency foreign exchange pair conversion rates",
    "tax_lot_report": "Generate cost basis and realized capital gains tax report",
    INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence to match any supported API function",
}

T09_ARMS: dict[FramingArm, dict[str, str]] = {
    "neutral": T09_FUNCTIONS_NEUTRAL,
    "verbose": {
        k: f"Deterministic programmatic API function invocation to {v}" if k != INSUFFICIENT_EVIDENCE_ID else "Insufficient evidence or general text query not matching any typed API signature"
        for k, v in T09_FUNCTIONS_NEUTRAL.items()
    },
    "banking_framed": {
        k: f"Financial services API endpoint to {v}" if k != INSUFFICIENT_EVIDENCE_ID else "Insufficient evidence for financial services API function execution"
        for k, v in T09_FUNCTIONS_NEUTRAL.items()
    },
}

# ---------------------------------------------------------------------------
# Task 10: Agent skill selection (20 skills + abstention = 21, authored fixture)
# ---------------------------------------------------------------------------
T10_SKILLS_NEUTRAL = {
    "apple_notes": "Manage Apple Notes via memo CLI: create, search, edit notes",
    "apple_reminders": "Apple Reminders via remindctl: add, list, complete tasks",
    "findmy": "Track Apple devices and AirTags via FindMy on macOS",
    "imessage": "Send and receive iMessages and SMS via imsg CLI",
    "pptx_author": "Build PowerPoint presentation decks with python-pptx",
    "powerpoint": "Read, inspect, and modify existing slide decks and templates",
    "chroma": "Vector embedding database for semantic document search and RAG",
    "xurl": "X and Twitter API via xurl CLI: search posts, post tweet, send DM",
    "computer_use": "Desktop automation: control mouse cursor, click buttons, type text",
    "openhands": "Delegate software coding tasks to OpenHands agent CLI",
    "git_workflow": "Git version control operations: create branch, commit, resolve merge",
    "sqlite_query": "Query and inspect relational SQLite databases",
    "pdf_extract": "Extract text, tables, and metadata from PDF files",
    "docker_manager": "Manage Docker containers, inspect logs, and build images",
    "web_scraper": "Scrape structured data from web pages and bypass anti-bot",
    "github_issue": "Create, triage, label, and comment on GitHub issues",
    "slack_notify": "Send formatted messages and alerts to Slack channels",
    "audio_transcribe": "Transcribe voice memos and audio recordings with Whisper",
    "code_reviewer": "Perform adversarial security and standards code review on diffs",
    "weather_forecast": "Fetch real-time meteorological forecasts and radar data",
    INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence to select a specialized agent skill",
}

T10_ARMS: dict[FramingArm, dict[str, str]] = {
    "neutral": T10_SKILLS_NEUTRAL,
    "verbose": {
        k: f"Specialized AI agent tool skill to {v}" if k != INSUFFICIENT_EVIDENCE_ID else "Insufficient evidence or general conversational query that requires no specialized skill"
        for k, v in T10_SKILLS_NEUTRAL.items()
    },
    "banking_framed": {
        k: f"Enterprise productivity and integration skill to {v}" if k != INSUFFICIENT_EVIDENCE_ID else "Insufficient evidence to invoke enterprise banking workspace skill"
        for k, v in T10_SKILLS_NEUTRAL.items()
    },
}

# ---------------------------------------------------------------------------
# Task 11: Passage relevance re-ranking (10 options)
# ---------------------------------------------------------------------------
T11_ARMS: dict[FramingArm, dict[str, str]] = {
    "neutral": {
        "passage_0": "Candidate passage 1 text",
        "passage_1": "Candidate passage 2 text",
        "passage_2": "Candidate passage 3 text",
        "passage_3": "Candidate passage 4 text",
        "passage_4": "Candidate passage 5 text",
        "passage_5": "Candidate passage 6 text",
        "passage_6": "Candidate passage 7 text",
        "passage_7": "Candidate passage 8 text",
        "passage_8": "Candidate passage 9 text",
        INSUFFICIENT_EVIDENCE_ID: "None of the candidate passages contain the answer",
    },
    "verbose": {
        f"passage_{i}": f"Candidate document passage {i+1} providing factual evidence"
        for i in range(9)
    } | {INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence across all candidate passages to answer the query"},
    "banking_framed": {
        f"passage_{i}": f"Candidate regulatory compliance disclosure {i+1}"
        for i in range(9)
    } | {INSUFFICIENT_EVIDENCE_ID: "Insufficient regulatory evidence in candidate disclosures"},
}

# ---------------------------------------------------------------------------
# Task 12: Prompt-injection screening in RAG (3 options)
# ---------------------------------------------------------------------------
T12_ARMS: dict[FramingArm, dict[str, str]] = {
    "neutral": {
        "clean_context": "Clean factual document context safe to pass to model",
        "prompt_injection": "Adversarial prompt injection attempt disguised in text",
        INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence or uninformative background noise",
    },
    "verbose": {
        "clean_context": "Legitimate, clean factual knowledge passage safe for LLM context retrieval and generation",
        "prompt_injection": "Malicious indirect prompt injection or jailbreak payload attempting to hijack downstream model execution",
        INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence or low-relevance uninformative text",
    },
    "banking_framed": {
        "clean_context": "Verified genuine banking policy documentation safe for customer query answering",
        "prompt_injection": "Adversarial prompt injection payload attempting to manipulate banking assistant output or leak customer data",
        INSUFFICIENT_EVIDENCE_ID: "Insufficient evidence or ambiguous non-banking content",
    },
}

# ---------------------------------------------------------------------------
# Master Task Registry
# ---------------------------------------------------------------------------
TASK_REGISTRY: dict[str, TaskSpec] = {
    "T01": TaskSpec(
        id="T01",
        name="Phishing vs Legitimate Email",
        cookbook="patterns_confidence_routing",
        source="ealvaradob/phishing-dataset, SetFit/enron_spam",
        k_cardinality=3,
        demo_value="High",
        description="Classify whether an incoming email is a fraudulent phishing attempt, legitimate correspondence, or requires abstention.",
        provenance="public",
        provenance_details="HuggingFace ealvaradob/phishing-dataset and SetFit/enron_spam public benchmarks",
        default_question="Classify whether this email is a fraudulent phishing attempt, legitimate correspondence, or requires abstention.",
        candidate_arms=T01_ARMS,
    ),
    "T02": TaskSpec(
        id="T02",
        name="Jailbreak vs Benign Prompt",
        cookbook="llm_guardrails",
        source="TrustAIRLab/in-the-wild-jailbreak-prompts, tatsu-lab/alpaca",
        k_cardinality=3,
        demo_value="High",
        description="Evaluate whether a user prompt contains adversarial jailbreak intent or is benign instruction.",
        provenance="public",
        provenance_details="TrustAIRLab in-the-wild jailbreak prompts and Stanford Alpaca instruction dataset",
        default_question="Evaluate whether this prompt is an adversarial jailbreak attempt, a benign user query, or insufficient evidence.",
        candidate_arms=T02_ARMS,
    ),
    "T03": TaskSpec(
        id="T03",
        name="Hazard Severity Routing",
        cookbook="llm_guardrails",
        source="TrustAIRLab/in-the-wild-jailbreak-prompts, tatsu-lab/alpaca",
        k_cardinality=5,
        demo_value="Medium",
        description="Classify prompt hazard severity level into benign, low-risk, jailbreak, or critical security threat.",
        provenance="public",
        provenance_details="Multi-tier severity mapping over in-the-wild jailbreaks and benign instructions",
        default_question="Classify the security hazard severity level of this prompt.",
        candidate_arms=T03_ARMS,
    ),
    "T04": TaskSpec(
        id="T04",
        name="Shopify Product Taxonomy (Flat Leaf)",
        cookbook="hierarchical_classification",
        source="Shopify product-taxonomy",
        k_cardinality=25,
        demo_value="Medium",
        description="Classify merchandise descriptions into 24 distinct Shopify retail product categories plus abstention.",
        provenance="public",
        provenance_details="Shopify official public open-source product taxonomy",
        default_question="Which product category best matches this merchandise?",
        candidate_arms=T04_ARMS,
    ),
    "T05": TaskSpec(
        id="T05",
        name="Shopify Taxonomy (Hierarchical Beam)",
        cookbook="hierarchical_classification",
        source="Shopify product-taxonomy",
        k_cardinality=7,
        demo_value="High",
        description="Top-level division routing for hierarchical beam search classification across deep catalog trees.",
        provenance="public",
        provenance_details="Shopify official product taxonomy division clustering",
        default_question="Which top-level merchandise division does this product belong to?",
        candidate_arms=T05_ARMS,
    ),
    "T06": TaskSpec(
        id="T06",
        name="Banking77 In-Domain Control",
        cookbook="banking77_intent_routing",
        source="PolyAI Banking77 (local test set)",
        k_cardinality=5,
        demo_value="Baseline",
        description="In-domain control benchmark evaluating customer banking intent classification across card lifecycle operations.",
        provenance="public",
        provenance_details="PolyAI Banking77 held-out test split under data/real_banking_test.jsonl",
        default_question="What is the customer's primary banking intent in this inquiry?",
        candidate_arms=T06_ARMS,
    ),
    "T07": TaskSpec(
        id="T07",
        name="CLINC150 Intent Domain Routing",
        cookbook="patterns_intent_routing",
        source="clinc/clinc_oos",
        k_cardinality=11,
        demo_value="High",
        description="Route multi-domain assistant queries across 10 functional domains plus out-of-scope abstention.",
        provenance="public",
        provenance_details="CLINC150 out-of-scope benchmark test partition",
        default_question="Which assistant service domain should handle this request?",
        candidate_arms=T07_ARMS,
    ),
    "T08": TaskSpec(
        id="T08",
        name="Smart Home Command Interpretation",
        cookbook="demos_smart_home",
        source="project-authored fixture",
        k_cardinality=8,
        demo_value="Highest",
        description="Interpret natural language smart home utterances into 7 discrete device control actions plus abstention.",
        provenance="authored",
        provenance_details="Project-authored fixture with deterministic ground truth, following openjev specification",
        default_question="Which home automation action does the user want executed?",
        candidate_arms=T08_ARMS,
    ),
    "T09": TaskSpec(
        id="T09",
        name="Function and Tool Routing",
        cookbook="function_calling",
        source="project-authored fixture",
        k_cardinality=21,
        demo_value="High",
        description="Map natural language trading and account commands to 20 typed API functions plus abstention.",
        provenance="authored",
        provenance_details="Project-authored fixture with deterministic type signatures and parameter constraints",
        default_question="Which API tool function should be invoked for this request?",
        candidate_arms=T09_ARMS,
    ),
    "T10": TaskSpec(
        id="T10",
        name="Agent Skill Selection",
        cookbook="skill_suggestion",
        source="Nous Hermes skill catalog",
        k_cardinality=21,
        demo_value="Medium",
        description="Select the optimal specialized agent skill from the 20-tool Hermes catalog for a given user request.",
        provenance="authored",
        provenance_details="Nous Research Hermes Agent skill roster and tool definitions",
        default_question="Which agent skill from the Hermes catalog should be loaded?",
        candidate_arms=T10_ARMS,
    ),
    "T11": TaskSpec(
        id="T11",
        name="Passage Relevance Re-ranking",
        cookbook="rerank_typesafe",
        source="mteb/scifact & legal QA",
        k_cardinality=10,
        demo_value="Medium",
        description="Re-rank 9 candidate retrieved passages to select the direct answering citation or abstain.",
        provenance="public",
        provenance_details="MTEB SciFact benchmark query-passage evaluation pairs",
        default_question="Which candidate passage directly answers the user search query?",
        candidate_arms=T11_ARMS,
    ),
    "T12": TaskSpec(
        id="T12",
        name="RAG Prompt-Injection Screening",
        cookbook="classifying_rag_passages",
        source="deepset/prompt-injections & clean docs",
        k_cardinality=3,
        demo_value="High",
        description="Screen retrieved knowledge chunks to detect adversarial prompt injection before answering model ingestion.",
        provenance="public",
        provenance_details="Deepset prompt-injections and verified clean knowledge passages",
        default_question="Classify whether this retrieved document is clean relevant context, a prompt injection payload, or noise.",
        candidate_arms=T12_ARMS,
    ),
}


def get_task_candidates(task_id: str, arm: FramingArm) -> list[dict[str, str]]:
    """Return candidates formatted for the specified task and framing arm."""
    spec = TASK_REGISTRY[task_id]
    arm_dict = spec.candidate_arms.get(arm, spec.candidate_arms["neutral"])
    return [{"id": cid, "description": desc} for cid, desc in arm_dict.items()]
