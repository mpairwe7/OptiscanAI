"""Stripe SDK wrapper — checkout sessions, billing portal, webhook verification."""

from __future__ import annotations

import logging
from typing import Any, Optional

import stripe

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


def _configure() -> None:
    if not settings.stripe.api_key:
        raise RuntimeError("Stripe not configured — set STRIPE__API_KEY")
    stripe.api_key = settings.stripe.api_key


def price_id_for(plan_code: str, billing_cycle: str) -> Optional[str]:
    s = settings.stripe
    table = {
        ("clinician", "monthly"): s.clinician_monthly_price_id,
        ("clinician", "annual"): s.clinician_annual_price_id,
        ("practice", "monthly"): s.practice_monthly_price_id,
        ("practice", "annual"): s.practice_annual_price_id,
    }
    return table.get((plan_code, billing_cycle)) or None


async def create_checkout_session(
    *,
    plan_code: str,
    billing_cycle: str,
    customer_email: str,
    organization_id: str,
    subscription_id: Optional[str],
    user_id: str,
    idempotency_key: str,
    existing_customer_id: Optional[str] = None,
) -> stripe.checkout.Session:
    _configure()
    price_id = price_id_for(plan_code, billing_cycle)
    if not price_id:
        raise ValueError(f"No Stripe price configured for {plan_code}/{billing_cycle}")

    kwargs: dict[str, Any] = {
        "mode": "subscription",
        "line_items": [{"price": price_id, "quantity": 1}],
        "success_url": settings.stripe.success_url,
        "cancel_url": settings.stripe.cancel_url,
        "allow_promotion_codes": True,
        "metadata": {
            "organization_id": organization_id,
            "plan_code": plan_code,
            "billing_cycle": billing_cycle,
            "user_id": user_id,
        },
        "subscription_data": {
            "metadata": {
                "organization_id": organization_id,
                "plan_code": plan_code,
                "billing_cycle": billing_cycle,
            },
        },
        "client_reference_id": organization_id,
    }
    if existing_customer_id:
        kwargs["customer"] = existing_customer_id
    else:
        kwargs["customer_email"] = customer_email
        kwargs["customer_creation"] = "always"

    return stripe.checkout.Session.create(**kwargs, idempotency_key=idempotency_key)


async def create_portal_session(*, customer_id: str) -> stripe.billing_portal.Session:
    _configure()
    return stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=settings.stripe.portal_return_url,
    )


def verify_webhook(payload: bytes, signature: str) -> stripe.Event:
    _configure()
    if not settings.stripe.webhook_secret:
        raise RuntimeError("Stripe webhook secret not configured — set STRIPE__WEBHOOK_SECRET")
    return stripe.Webhook.construct_event(
        payload=payload,
        sig_header=signature,
        secret=settings.stripe.webhook_secret,
    )


async def retrieve_subscription(stripe_subscription_id: str) -> stripe.Subscription:
    _configure()
    return stripe.Subscription.retrieve(stripe_subscription_id)


def extra_seat_price_id(billing_cycle: str) -> Optional[str]:
    s = settings.stripe
    if billing_cycle == "annual":
        return s.practice_extra_seat_annual_price_id or None
    return s.practice_extra_seat_monthly_price_id or None


async def set_seat_quantity(
    *,
    stripe_subscription_id: str,
    seat_item_id: Optional[str],
    new_quantity: int,
    billing_cycle: str,
) -> tuple[stripe.Subscription, str]:
    """Add, update, or remove the extra-seat line on a Stripe subscription.

    Returns ``(subscription, seat_item_id)``. If ``new_quantity == 0`` the seat
    item is deleted and the returned ``seat_item_id`` is "".
    """
    _configure()
    price_id = extra_seat_price_id(billing_cycle)
    if not price_id:
        raise RuntimeError(f"No Stripe extra-seat price configured for cycle={billing_cycle}")

    if seat_item_id and new_quantity == 0:
        # Remove the item entirely
        stripe.SubscriptionItem.delete(seat_item_id, proration_behavior="create_prorations")
        sub = stripe.Subscription.retrieve(stripe_subscription_id)
        return sub, ""

    if seat_item_id:
        # Update existing item quantity
        item = stripe.SubscriptionItem.modify(
            seat_item_id,
            quantity=new_quantity,
            proration_behavior="create_prorations",
        )
    else:
        # Create the seat item on the existing subscription
        item = stripe.SubscriptionItem.create(
            subscription=stripe_subscription_id,
            price=price_id,
            quantity=new_quantity,
            proration_behavior="create_prorations",
        )

    sub = stripe.Subscription.retrieve(stripe_subscription_id)
    return sub, item.id
