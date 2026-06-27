"""Manual test runner for inbox triage.

Usage:
    Terminal 1:  make serve
    Terminal 2:  py -3 test_triage.py              # auto mode (approve e-001 only)
                 py -3 test_triage.py --deny-all    # deny every action
                 py -3 test_triage.py --interactive # prompt y/n for each action
                 py -3 test_triage.py --use-ai --interactive  # classify with Anthropic
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

from src.triage_skill import (
    API_BASE_URL,
    READ_TOKEN,
    TriageClient,
    classify_email,
    console_approver,
    triage_inbox,
)

# Stand-in labels until classify_email is wired up.
STUB_LABELS = {
    "e-001": "billing",
    "e-002": "bug_report",
    "e-003": "sales_lead",
    "e-004": "spam",
    "e-005": "billing",
    "e-006": "bug_report",
    "e-007": "spam",
    "e-008": "sales_lead",
}


def stub_classifier(email: dict) -> str:
    email_id = email["id"]
    if email_id not in STUB_LABELS:
        raise KeyError(f"No stub label for {email_id!r}")
    return STUB_LABELS[email_id]


def approve_first_billing_only(email: dict, action, label: str = "") -> bool:
    """Auto-approver: only the first billing reply (e-001) is approved."""
    print(f"[{email['id']}] classified as: {label or 'unknown'}")
    approved = email["id"] == "e-001" and action.kind == "send_reply"
    print(f"  -> {'approved' if approved else 'denied'} (auto)")
    return approved


def deny_all(_email: dict, _action, label: str = "") -> bool:
    print(f"[{_email['id']}] classified as: {label or 'unknown'}")
    print("  -> denied (auto)")
    return False


def check_api() -> None:
    try:
        response = httpx.get(
            f"{API_BASE_URL}/inbox",
            headers={"Authorization": f"Bearer {READ_TOKEN}"},
            timeout=5,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Mock API not reachable at {API_BASE_URL}: {exc}", file=sys.stderr)
        print("Start it first with: make serve", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test inbox triage with a stub classifier.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt y/n for each proposed action (uses console_approver).",
    )
    group.add_argument(
        "--deny-all",
        action="store_true",
        help="Deny every proposed action (no side effects).",
    )
    parser.add_argument(
        "--use-ai",
        action="store_true",
        help="Use Anthropic classify_email instead of stub labels.",
    )
    args = parser.parse_args()

    if args.interactive:
        approver = console_approver
    elif args.deny_all:
        approver = deny_all
    else:
        approver = approve_first_billing_only

    classifier = classify_email if args.use_ai else stub_classifier

    check_api()

    mode = "AI classifier" if args.use_ai else "stub classifier"
    print("=" * 50)
    print(f"INBOX TRIAGE TEST ({mode})")
    print("=" * 50)

    read_client = TriageClient(API_BASE_URL, READ_TOKEN)
    results = triage_inbox(read_client, approver, classifier=classifier)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for result in results:
        print(f"  {result.email_id}: {result.label} ({len(result.actions)} action(s))")

    audit = httpx.get(f"{API_BASE_URL}/_audit").json()
    print("\n" + "=" * 50)
    print("AUDIT")
    print("=" * 50)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
