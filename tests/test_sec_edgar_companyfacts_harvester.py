"""Tests for the SEC EDGAR companyfacts harvester."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from renquant_base_data.sec_edgar_companyfacts_harvester import (
    CANONICAL_CONCEPTS,
    extract_facts,
    harvest,
    load_completed_tickers,
    parse_ticker_cik_map,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TICKER_CIK_JSON = {
    "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": "789019", "ticker": "MSFT", "title": "Microsoft Corporation"},
    "2": {"cik_str": "1018724", "ticker": "AMZN", "title": "Amazon.com Inc."},
}


def _revenue_entry(val, accn, fy, fp, form, filed, end, start=None):
    entry = {
        "val": val, "accn": accn, "fy": fy, "fp": fp,
        "form": form, "filed": filed, "end": end,
    }
    if start is not None:
        entry["start"] = start
    return entry


SAMPLE_COMPANY_FACTS = {
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {
        "us-gaap": {
            "Revenues": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        _revenue_entry(
                            94836000000, "0000320193-22-000007", 2022, "Q1",
                            "10-Q", "2022-01-28", "2021-12-25", "2021-09-26",
                        ),
                        _revenue_entry(
                            365817000000, "0000320193-21-000105", 2021, "FY",
                            "10-K", "2021-10-29", "2021-09-25", "2020-09-27",
                        ),
                        _revenue_entry(
                            123456000000, "0000320193-22-000099", 2022, "Q2",
                            "8-K", "2022-04-28", "2022-03-26",
                        ),
                    ]
                },
            },
            "NetIncomeLoss": {
                "label": "Net Income (Loss)",
                "units": {
                    "USD": [
                        {
                            "val": 34630000000,
                            "accn": "0000320193-22-000007",
                            "fy": 2022,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2022-01-28",
                            "end": "2021-12-25",
                            "start": "2021-09-26",
                        },
                    ]
                },
            },
            "EarningsPerShareDiluted": {
                "label": "Earnings Per Share, Diluted",
                "units": {
                    "USD/shares": [
                        {
                            "val": 2.10,
                            "accn": "0000320193-22-000007",
                            "fy": 2022,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2022-01-28",
                            "end": "2021-12-25",
                            "start": "2021-09-26",
                        },
                    ]
                },
            },
            "Assets": {
                "label": "Assets",
                "units": {
                    "USD": [
                        {
                            "val": 381191000000,
                            "accn": "0000320193-22-000007",
                            "fy": 2022,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2022-01-28",
                            "end": "2021-12-25",
                        },
                    ]
                },
            },
        }
    },
}

# A second issuer that only reports revenue under the ASC-606-era tag, never
# the plain ``Revenues`` tag — this is exactly the case the old (buggy)
# harvester mapped to a DIFFERENT output field name (``revenue_alt``).
SAMPLE_COMPANY_FACTS_ASC606 = {
    "cik": 1121788,
    "entityName": "Garmin Ltd.",
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "label": "Revenue from Contract with Customer, Excluding Assessed Tax",
                "units": {
                    "USD": [
                        {
                            "val": 1500000000,
                            "accn": "0001121788-23-000010",
                            "fy": 2023,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2023-05-10",
                            "end": "2023-03-25",
                            "start": "2022-12-25",
                        },
                    ]
                },
            },
        }
    },
}


# ---------------------------------------------------------------------------
# Tests: CIK mapping
# ---------------------------------------------------------------------------


class TestTickerCikMap:
    def test_parse_basic(self):
        result = parse_ticker_cik_map(SAMPLE_TICKER_CIK_JSON)
        assert result["AAPL"] == 320193
        assert result["MSFT"] == 789019
        assert result["AMZN"] == 1018724

    def test_parse_uppercases(self):
        data = {"0": {"cik_str": "123", "ticker": "grmn", "title": "Garmin"}}
        result = parse_ticker_cik_map(data)
        assert "GRMN" in result

    def test_parse_skips_missing_fields(self):
        data = {
            "0": {"cik_str": "123", "ticker": "AAPL", "title": "Apple"},
            "1": {"cik_str": "456", "title": "No Ticker"},
            "2": {"ticker": "NOCE", "title": "No CIK"},
        }
        result = parse_ticker_cik_map(data)
        assert len(result) == 1
        assert "AAPL" in result

    def test_parse_empty(self):
        assert parse_ticker_cik_map({}) == {}


# ---------------------------------------------------------------------------
# Tests: fact extraction + canonical field normalization
# ---------------------------------------------------------------------------


class TestExtractFacts:
    def test_extracts_all_fields(self):
        records = extract_facts("AAPL", SAMPLE_COMPANY_FACTS)
        fields = {r["field"] for r in records}
        assert "revenue" in fields
        assert "net_income" in fields
        assert "eps_diluted" in fields
        assert "total_assets" in fields

    def test_preserves_filed_date(self):
        records = extract_facts("AAPL", SAMPLE_COMPANY_FACTS)
        revenue_q1 = [
            r for r in records if r["field"] == "revenue" and r["fiscal_period"] == "Q1"
        ]
        assert len(revenue_q1) == 1
        assert revenue_q1[0]["filed_date"] == "2022-01-28"

    def test_filters_10k_10q_only(self):
        records = extract_facts("AAPL", SAMPLE_COMPANY_FACTS)
        forms = {r["form"] for r in records}
        assert forms <= {"10-K", "10-Q"}
        assert "8-K" not in forms

    def test_revenue_count(self):
        records = extract_facts("AAPL", SAMPLE_COMPANY_FACTS)
        revenues = [r for r in records if r["field"] == "revenue"]
        assert len(revenues) == 2  # Q1 (10-Q) + FY (10-K); 8-K filtered out

    def test_eps_uses_usd_per_shares(self):
        records = extract_facts("AAPL", SAMPLE_COMPANY_FACTS)
        eps = [r for r in records if r["field"] == "eps_diluted"]
        assert len(eps) == 1
        assert eps[0]["value"] == 2.10

    def test_record_structure(self):
        records = extract_facts("AAPL", SAMPLE_COMPANY_FACTS)
        rec = records[0]
        required_keys = {
            "ticker", "field", "xbrl_tag", "value", "filed_date",
            "period_end", "fiscal_year", "fiscal_period", "form",
            "accession_number", "source",
        }
        assert required_keys <= set(rec.keys())
        assert rec["source"] == "sec_edgar_xbrl"
        assert rec["ticker"] == "AAPL"

    def test_empty_facts(self):
        empty = {"facts": {"us-gaap": {}}}
        records = extract_facts("XYZ", empty)
        assert records == []

    def test_missing_us_gaap(self):
        no_gaap = {"facts": {}}
        records = extract_facts("XYZ", no_gaap)
        assert records == []

    def test_asc606_revenue_tag_normalizes_to_same_canonical_field(self):
        """The core field-mapping fix: RevenueFromContractWithCustomer*
        must land under the SAME 'revenue' field as the plain 'Revenues' tag,
        not a differently-named 'revenue_alt' field."""
        records = extract_facts("GRMN", SAMPLE_COMPANY_FACTS_ASC606)
        fields = {r["field"] for r in records}
        assert fields == {"revenue"}
        assert "revenue_alt" not in fields

    def test_asc606_tag_preserved_as_provenance(self):
        records = extract_facts("GRMN", SAMPLE_COMPANY_FACTS_ASC606)
        assert len(records) == 1
        assert records[0]["field"] == "revenue"
        assert records[0]["xbrl_tag"] == (
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        )

    def test_plain_revenues_tag_also_canonical(self):
        records = extract_facts("AAPL", SAMPLE_COMPANY_FACTS)
        revenue_recs = [r for r in records if r["field"] == "revenue"]
        assert all(r["xbrl_tag"] == "us-gaap:Revenues" for r in revenue_recs)

    def test_all_known_revenue_tags_map_to_revenue(self):
        assert set(CANONICAL_CONCEPTS["revenue"]) >= {
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
        }


# ---------------------------------------------------------------------------
# Tests: resumability (marker-based, not any-record-present)
# ---------------------------------------------------------------------------


class TestResumability:
    def test_load_completed_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.touch()
        assert load_completed_tickers(p) == set()

    def test_load_completed_nonexistent(self, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        assert load_completed_tickers(p) == set()

    def test_load_completed_requires_marker(self, tmp_path):
        """A ticker with fact records but NO completion marker must NOT
        count as done — this is the resumability fix: the old code treated
        any record's presence as proof of completion, so a crash mid-ticker
        (partial records written, marker never reached) would cause a rerun
        to silently skip re-harvesting it."""
        p = tmp_path / "out.jsonl"
        p.write_text(
            json.dumps({"ticker": "AAPL", "field": "revenue"}) + "\n"
            + json.dumps({"ticker": "AAPL", "field": "net_income"}) + "\n"
        )
        result = load_completed_tickers(p)
        assert result == set()

    def test_load_completed_with_marker(self, tmp_path):
        p = tmp_path / "out.jsonl"
        p.write_text(
            json.dumps({"ticker": "AAPL", "field": "revenue"}) + "\n"
            + json.dumps({"ticker": "AAPL", "_harvest_complete": True}) + "\n"
            + json.dumps({"ticker": "GRMN", "field": "revenue"}) + "\n"
            + json.dumps({"ticker": "GRMN", "_harvest_complete": True}) + "\n"
        )
        result = load_completed_tickers(p)
        assert result == {"AAPL", "GRMN"}

    def test_load_completed_partial_ticker_excluded(self, tmp_path):
        """AAPL has its marker (done); GRMN has records but no marker
        (interrupted mid-write) — GRMN must be re-harvested on rerun."""
        p = tmp_path / "out.jsonl"
        p.write_text(
            json.dumps({"ticker": "AAPL", "field": "revenue"}) + "\n"
            + json.dumps({"ticker": "AAPL", "_harvest_complete": True}) + "\n"
            + json.dumps({"ticker": "GRMN", "field": "revenue"}) + "\n"
        )
        result = load_completed_tickers(p)
        assert result == {"AAPL"}

    def test_load_completed_handles_bad_json(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text(
            '{"ticker": "AAPL"}\nnot json\n'
            '{"ticker": "AAPL", "_harvest_complete": true}\n'
        )
        result = load_completed_tickers(p)
        assert result == {"AAPL"}


# ---------------------------------------------------------------------------
# Tests: harvest integration (mocked HTTP)
# ---------------------------------------------------------------------------


class TestHarvest:
    def _mock_session(self, facts=SAMPLE_COMPANY_FACTS):
        session = MagicMock()

        def get_side_effect(url, **kwargs):
            resp = MagicMock()
            if "companyfacts" in url:
                resp.status_code = 200
                resp.json.return_value = facts
            else:
                resp.status_code = 200
                resp.json.return_value = SAMPLE_TICKER_CIK_JSON
            resp.raise_for_status = MagicMock()
            return resp

        session.get.side_effect = get_side_effect
        return session

    @patch("renquant_base_data.sec_edgar_companyfacts_harvester.time.sleep")
    def test_harvest_basic(self, mock_sleep):
        session = self._mock_session()
        cik_map = {"AAPL": 320193}
        records = harvest(["AAPL"], session=session, ticker_cik_map=cik_map)
        assert len(records) > 0
        assert all(r["ticker"] == "AAPL" for r in records)

    @patch("renquant_base_data.sec_edgar_companyfacts_harvester.time.sleep")
    def test_harvest_writes_jsonl_with_marker(self, mock_sleep, tmp_path):
        session = self._mock_session()
        cik_map = {"AAPL": 320193}
        output = tmp_path / "out.jsonl"
        harvest(["AAPL"], output, session=session, ticker_cik_map=cik_map)
        lines = output.read_text().strip().splitlines()
        assert len(lines) > 0
        recs = [json.loads(line) for line in lines]
        assert any(r.get("_harvest_complete") for r in recs)
        fact_recs = [r for r in recs if "filed_date" in r]
        assert all(r["ticker"] == "AAPL" for r in fact_recs)

    @patch("renquant_base_data.sec_edgar_companyfacts_harvester.time.sleep")
    def test_harvest_skips_only_marked_complete(self, mock_sleep, tmp_path):
        session = self._mock_session()
        cik_map = {"AAPL": 320193, "GRMN": 1121788}
        output = tmp_path / "out.jsonl"
        output.write_text(
            json.dumps({"ticker": "AAPL", "field": "revenue"}) + "\n"
            + json.dumps({"ticker": "AAPL", "_harvest_complete": True}) + "\n"
        )
        records = harvest(
            ["AAPL", "GRMN"], output, session=session, ticker_cik_map=cik_map
        )
        assert all(r["ticker"] == "GRMN" for r in records)

    @patch("renquant_base_data.sec_edgar_companyfacts_harvester.time.sleep")
    def test_harvest_rerun_after_partial_crash_reharvests_ticker(self, mock_sleep, tmp_path):
        """Simulates a crash: AAPL's fact records were written but the
        process died before the completion marker landed. A rerun must
        re-harvest AAPL rather than skip it."""
        session = self._mock_session()
        cik_map = {"AAPL": 320193}
        output = tmp_path / "out.jsonl"
        output.write_text(
            json.dumps({"ticker": "AAPL", "field": "revenue"}) + "\n"
        )  # no marker: simulates interrupted write
        records = harvest(["AAPL"], output, session=session, ticker_cik_map=cik_map)
        assert len(records) > 0
        assert all(r["ticker"] == "AAPL" for r in records)

    @patch("renquant_base_data.sec_edgar_companyfacts_harvester.time.sleep")
    def test_harvest_rerun_after_partial_crash_does_not_duplicate_facts(
        self, mock_sleep, tmp_path
    ):
        """A crash that leaves AAPL's fact lines on disk WITHOUT its marker
        (simulated pre-existing state) must not cause a rerun to duplicate
        those facts on top of the surviving partial write — the final file
        must contain each fact record for the ticker exactly once."""
        session = self._mock_session()
        cik_map = {"AAPL": 320193}
        output = tmp_path / "out.jsonl"
        partial_records = extract_facts("AAPL", SAMPLE_COMPANY_FACTS)
        output.write_text(
            "".join(
                json.dumps(r, sort_keys=True) + "\n" for r in partial_records
            )
        )  # facts persisted, no marker: simulates interrupted write

        harvest(["AAPL"], output, session=session, ticker_cik_map=cik_map)

        lines = output.read_text().strip().splitlines()
        recs = [json.loads(line) for line in lines]
        fact_recs = [r for r in recs if "filed_date" in r]
        marker_recs = [r for r in recs if r.get("_harvest_complete")]

        # Exactly one marker for AAPL, and no fact record appears more than
        # once (dedup key: field + period_end + accession_number).
        assert len(marker_recs) == 1
        seen = set()
        for r in fact_recs:
            key = (r["field"], r["period_end"], r["accession_number"])
            assert key not in seen, f"duplicate fact record: {key}"
            seen.add(key)
        assert len(fact_recs) == len(partial_records)

    @patch("renquant_base_data.sec_edgar_companyfacts_harvester.time.sleep")
    def test_harvest_missing_cik(self, mock_sleep):
        session = self._mock_session()
        cik_map = {}
        records = harvest(["ZZZZZ"], session=session, ticker_cik_map=cik_map)
        assert records == []

    @patch("renquant_base_data.sec_edgar_companyfacts_harvester.time.sleep")
    def test_harvest_404_graceful(self, mock_sleep):
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 404
        session.get.return_value = resp
        cik_map = {"AAPL": 320193}
        records = harvest(["AAPL"], session=session, ticker_cik_map=cik_map)
        assert records == []

    @patch("renquant_base_data.sec_edgar_companyfacts_harvester.time.sleep")
    def test_harvest_asc606_issuer_writes_canonical_revenue(self, mock_sleep, tmp_path):
        session = self._mock_session(facts=SAMPLE_COMPANY_FACTS_ASC606)
        cik_map = {"GRMN": 1121788}
        output = tmp_path / "out.jsonl"
        records = harvest(["GRMN"], output, session=session, ticker_cik_map=cik_map)
        assert len(records) == 1
        assert records[0]["field"] == "revenue"


# ---------------------------------------------------------------------------
# Tests: output format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_jsonl_roundtrip(self, tmp_path):
        records = extract_facts("AAPL", SAMPLE_COMPANY_FACTS)
        output = tmp_path / "test.jsonl"
        with open(output, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

        loaded = []
        with open(output) as f:
            for line in f:
                loaded.append(json.loads(line))

        assert len(loaded) == len(records)
        for orig, read in zip(records, loaded):
            assert orig["ticker"] == read["ticker"]
            assert orig["filed_date"] == read["filed_date"]
            assert orig["value"] == read["value"]
