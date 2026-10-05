"""The access model: roles, levels, departments, tenants, and the SQL it generates."""

import pytest

from app.ingestion.chunking import chunk_document
from app.ingestion.loader import DocumentFormatError, parse_document
from app.security.access import ACCESS_LEVELS, AccessScope, scope_for

DOC = (
    '---\nname: a\ntitle: A\nversion: "1.0"\ncategory: c\nstatus: current\n{extra}---\n\n'
    "## S\nText.\n"
)


def scope(
    roles: list[str], departments: list[str] | None = None, tenant: str = "t1"
) -> AccessScope:
    result = scope_for(roles, tenant, departments or [])
    assert result is not None
    return result


# --- scopes from roles -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("roles", "level"),
    [
        (["employee"], "internal"),
        (["underwriter"], "restricted"),
        (["compliance"], "restricted"),
        (["admin"], "confidential"),
        (["employee", "underwriter"], "restricted"),
        (["employee", "admin"], "confidential"),
    ],
)
def test_a_role_sets_the_highest_readable_level(roles: list[str], level: str) -> None:
    assert scope(roles).max_level == level


def test_unknown_roles_are_ignored_not_trusted() -> None:
    assert scope_for(["superuser", "root", "Admin "], "t1", []) is None
    granted = scope(["superuser", "employee"])
    assert granted.roles == ("employee",)
    assert granted.max_level == "internal"


def test_no_recognised_role_means_no_scope() -> None:
    assert scope_for([], "t1", []) is None


def test_only_admins_belong_to_every_department() -> None:
    assert scope(["admin"]).all_departments is True
    assert scope(["underwriter"], ["underwriting"]).all_departments is False


def test_departments_are_normalised() -> None:
    assert scope(["underwriter"], [" Underwriting ", ""]).departments == frozenset({"underwriting"})


# --- the rule ----------------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["public", "internal"])
@pytest.mark.parametrize("department", ["general", "underwriting", "legal"])
def test_open_levels_are_readable_across_departments_within_the_tenant(
    level: str, department: str
) -> None:
    assert scope(["employee"]).permits("t1", level, department)


def test_an_employee_cannot_read_restricted_or_confidential_documents() -> None:
    employee = scope(["employee"], ["underwriting"])
    assert not employee.permits("t1", "restricted", "underwriting")
    assert not employee.permits("t1", "confidential", "underwriting")


def test_restricted_documents_need_the_right_department() -> None:
    underwriter = scope(["underwriter"], ["underwriting"])
    assert underwriter.permits("t1", "restricted", "underwriting")
    assert not underwriter.permits("t1", "restricted", "compliance")
    assert not scope(["underwriter"]).permits("t1", "restricted", "underwriting")  # no department


def test_restricted_roles_still_cannot_read_confidential() -> None:
    assert not scope(["compliance"], ["compliance"]).permits("t1", "confidential", "compliance")


def test_admins_read_everything_in_their_own_tenant_only() -> None:
    admin = scope(["admin"])
    assert admin.permits("t1", "confidential", "legal")
    assert not admin.permits("t2", "public", "general")


def test_other_tenants_are_never_readable() -> None:
    for level in ACCESS_LEVELS:
        assert not scope(["admin"]).permits("someone-else", level, "general")


def test_an_unlabelled_or_unknown_level_fails_closed() -> None:
    admin = scope(["admin"])
    assert not admin.permits("t1", "", "general")
    assert not admin.permits("t1", "top-secret", "general")


# --- the SQL -----------------------------------------------------------------------------------


def test_the_chunk_predicate_binds_every_value() -> None:
    hostile = scope(["underwriter"], ["x'; DROP TABLE chunks; --"], tenant="t'; DROP TABLE x; --")
    sql, params = hostile.chunk_clause("c.")
    assert "DROP" not in sql
    assert "%(acc_tenant)s" in sql
    assert params["acc_tenant"] == "t'; DROP TABLE x; --"
    assert params["acc_depts"] == ["x'; drop table chunks; --"]
    assert "c.metadata" in sql


def test_the_document_predicate_uses_columns_and_the_same_parameters() -> None:
    sql, params = scope(["employee"]).document_clause("d.")
    assert "d.tenant_id" in sql
    assert "d.access_level" in sql
    assert "d.department" in sql
    assert set(params) == {"acc_tenant", "acc_rank", "acc_open", "acc_all", "acc_depts"}


def test_a_missing_rank_counts_as_above_every_clearance() -> None:
    sql, _ = scope(["admin"]).chunk_clause()
    assert "99" in sql  # the fallback rank for an unlabelled chunk


# --- document labels at ingestion --------------------------------------------------------------


def test_unlabelled_documents_default_to_internal_in_the_default_tenant() -> None:
    doc = parse_document(DOC.format(extra=""), "a.md")
    assert (doc.tenant_id, doc.department, doc.access_level) == ("default", "general", "internal")


def test_labels_are_read_and_attached_to_every_chunk() -> None:
    extra = "tenant_id: acme\ndepartment: Underwriting\naccess_level: restricted\n"
    doc = parse_document(DOC.format(extra=extra), "a.md")
    assert (doc.tenant_id, doc.department, doc.access_level) == (
        "acme",
        "underwriting",
        "restricted",
    )
    for chunk in chunk_document(doc, 300, 60):
        assert chunk.metadata["tenant_id"] == "acme"
        assert chunk.metadata["department"] == "underwriting"
        assert chunk.metadata["access_level"] == "restricted"
        assert chunk.metadata["access_rank"] == "2"


@pytest.mark.parametrize(
    "extra",
    [
        "access_level: secret\n",
        "access_level: \n",
        "tenant_id: bad tenant!\n",
        "department: a/b\n",
        "tenant_id: " + "x" * 65 + "\n",
    ],
)
def test_invalid_labels_are_rejected(extra: str) -> None:
    with pytest.raises(DocumentFormatError):
        parse_document(DOC.format(extra=extra), "a.md")
