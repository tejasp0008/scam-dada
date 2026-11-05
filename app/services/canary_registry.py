"""Issuing canary identities and catching them resurfacing on a later call.

The detection is the interesting half and it is inherently cross-call: a leak means details
given to call A came back out of a caller's mouth on call B. So this needs the database,
which is why it is a service rather than part of `app.core.canary` - the generation and the
matching maths are pure and live there.

**Prior identities are loaded once, at the start of a call.** Matching on every turn against
a query would put a database round trip on the analysis path for no benefit: an identity
issued *during* this call cannot leak into it, so the set that matters is fixed when the call
begins.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.canary import CanaryMatch, find_matches, issue
from app.core.models import CanaryIdentity
from app.infra.models_sql import CanaryIdentity as CanaryRow
from app.infra.models_sql import CanaryMatch as MatchRow

logger = logging.getLogger(__name__)

#: How many previously issued identities a call checks itself against. Bounded because the
#: matching cost is linear and the newest leads are the ones a scammer is working from.
MAX_PRIOR_IDENTITIES = 500


class CanaryRegistry:
    """One call's canary identity, plus the earlier ones it watches for."""

    def __init__(
        self,
        call_id: str,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._call_id = call_id
        self._sessions = session_factory
        self._identity = issue(call_id)
        self._prior: list[tuple[str, CanaryIdentity]] = []
        self._seen: set[tuple[str, str]] = set()

    @property
    def identity(self) -> CanaryIdentity:
        """The fictional victim details this call will give out if asked."""
        return self._identity

    @property
    def prior_count(self) -> int:
        """How many earlier identities this call is watching for."""
        return len(self._prior)

    async def start(self) -> None:
        """Persist this call's identity and load the earlier ones. Never raises."""
        if self._sessions is None:
            return
        try:
            async with self._sessions() as session, session.begin():
                session.add(
                    CanaryRow(
                        id=self._identity.identity_id,
                        issued_call_id=self._call_id,
                        identity=self._identity.model_dump(mode="json"),
                    )
                )
        except Exception:
            # A call that cannot record its canary can still run; it just cannot be the
            # source of a future leak match.
            logger.exception("could not issue a canary identity", extra={"call_id": self._call_id})

        try:
            async with self._sessions() as session:
                rows = await session.execute(
                    select(CanaryRow)
                    .where(CanaryRow.issued_call_id != self._call_id)
                    .order_by(CanaryRow.issued_at.desc())
                    .limit(MAX_PRIOR_IDENTITIES)
                )
                for row in rows.scalars():
                    try:
                        self._prior.append(
                            (row.issued_call_id, CanaryIdentity.model_validate(row.identity))
                        )
                    except ValueError:
                        # An identity written by an older schema. Skipped rather than
                        # migrated: a leak match is a strong claim and it should rest on a
                        # record we can still parse exactly.
                        logger.info("skipping an unparseable canary row", extra={"id": row.id})
        except Exception:
            logger.exception("could not load prior canaries", extra={"call_id": self._call_id})

    def check(self, text: str) -> tuple[CanaryMatch, ...]:
        """Matches in this text that have not already been reported on this call.

        Deduplicated per call: a scammer who repeats a leaked handle three times is one
        leak, and three identical alerts would make the dashboard look broken.
        """
        fresh: list[CanaryMatch] = []
        for match in find_matches(text, self._prior):
            key = (match.identity_id, match.matched_field)
            if key not in self._seen:
                self._seen.add(key)
                fresh.append(match)
        return tuple(fresh)

    async def record(self, matches: Sequence[CanaryMatch]) -> None:
        """Persist detected leaks. Never raises."""
        if self._sessions is None or not matches:
            return
        try:
            async with self._sessions() as session, session.begin():
                for match in matches:
                    session.add(
                        MatchRow(
                            identity_id=match.identity_id,
                            matched_call_id=self._call_id,
                            matched_field=match.matched_field,
                            matched_text=match.matched_text,
                            score=match.score,
                        )
                    )
        except Exception:
            logger.exception("could not record canary matches", extra={"call_id": self._call_id})
