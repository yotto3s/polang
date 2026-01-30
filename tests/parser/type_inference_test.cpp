#include <gtest/gtest.h>

#include "parser/type_inference.hpp"

using namespace polang;

// ============== Substitution Tests ==============

TEST(SubstitutionTest, EmptySubstitution) {
  Substitution subst;
  EXPECT_EQ(subst.apply("i64"), "i64");
  EXPECT_EQ(subst.apply("$0"), "$0");
}

TEST(SubstitutionTest, BindAndApply) {
  Substitution subst;
  subst.bind("$0", "i64");
  EXPECT_EQ(subst.apply("$0"), "i64");
  EXPECT_EQ(subst.apply("$1"), "$1");
  EXPECT_EQ(subst.apply("i64"), "i64");
}

TEST(SubstitutionTest, TransitiveBind) {
  Substitution subst;
  subst.bind("$0", "$1");
  subst.bind("$1", "i64");
  EXPECT_EQ(subst.apply("$0"), "i64");
  EXPECT_EQ(subst.apply("$1"), "i64");
}

TEST(SubstitutionTest, Compose) {
  Substitution s1;
  s1.bind("$0", "$1");

  Substitution s2;
  s2.bind("$1", "f64");

  Substitution composed = s1.compose(s2);
  EXPECT_EQ(composed.apply("$0"), "f64");
  EXPECT_EQ(composed.apply("$1"), "f64");
}

TEST(SubstitutionTest, ComposeDoesNotOverwrite) {
  Substitution s1;
  s1.bind("$0", "i64");

  Substitution s2;
  s2.bind("$0", "f64");

  Substitution composed = s1.compose(s2);
  // s1's binding for $0 should take precedence (applied through s2)
  EXPECT_EQ(composed.apply("$0"), "i64");
}

TEST(SubstitutionTest, MultipleBindings) {
  Substitution subst;
  subst.bind("$0", "i64");
  subst.bind("$1", "bool");
  subst.bind("$2", "f64");
  EXPECT_EQ(subst.apply("$0"), "i64");
  EXPECT_EQ(subst.apply("$1"), "bool");
  EXPECT_EQ(subst.apply("$2"), "f64");
}

// ============== Unifier Tests ==============

TEST(UnifierTest, UnifySameConcreteTypes) {
  Unifier unifier;
  Substitution subst;
  EXPECT_TRUE(unifier.unify("i64", "i64", subst));
}

TEST(UnifierTest, UnifyDifferentConcreteTypesFails) {
  Unifier unifier;
  Substitution subst;
  EXPECT_FALSE(unifier.unify("i64", "f64", subst));
}

TEST(UnifierTest, UnifyVarWithConcrete) {
  Unifier unifier;
  Substitution subst;
  EXPECT_TRUE(unifier.unify("$0", "i64", subst));
  EXPECT_EQ(subst.apply("$0"), "i64");
}

TEST(UnifierTest, UnifyConcreteWithVar) {
  Unifier unifier;
  Substitution subst;
  EXPECT_TRUE(unifier.unify("i64", "$0", subst));
  EXPECT_EQ(subst.apply("$0"), "i64");
}

TEST(UnifierTest, UnifyTwoVars) {
  Unifier unifier;
  Substitution subst;
  EXPECT_TRUE(unifier.unify("$0", "$1", subst));
  // After unification, $0 and $1 should be equivalent
  std::string r0 = subst.apply("$0");
  std::string r1 = subst.apply("$1");
  EXPECT_EQ(r0, r1);
}

TEST(UnifierTest, UnifyChain) {
  Unifier unifier;
  Substitution subst;
  EXPECT_TRUE(unifier.unify("$0", "$1", subst));
  EXPECT_TRUE(unifier.unify("$1", "i64", subst));
  EXPECT_EQ(subst.apply("$0"), "i64");
  EXPECT_EQ(subst.apply("$1"), "i64");
}

TEST(UnifierTest, UnifyVarAlreadyBound) {
  Unifier unifier;
  Substitution subst;
  EXPECT_TRUE(unifier.unify("$0", "i64", subst));
  EXPECT_TRUE(unifier.unify("$0", "i64", subst));
}

TEST(UnifierTest, UnifyVarBoundConflict) {
  Unifier unifier;
  Substitution subst;
  EXPECT_TRUE(unifier.unify("$0", "i64", subst));
  EXPECT_FALSE(unifier.unify("$0", "f64", subst));
}

TEST(UnifierTest, UnifyGenericIntWithConcrete) {
  Unifier unifier;
  Substitution subst;
  EXPECT_TRUE(unifier.unify("{int}", "i32", subst));
}

TEST(UnifierTest, UnifyGenericIntWithFloat) {
  Unifier unifier;
  Substitution subst;
  EXPECT_FALSE(unifier.unify("{int}", "f64", subst));
}

TEST(UnifierTest, UnifyGenericFloatWithConcrete) {
  Unifier unifier;
  Substitution subst;
  EXPECT_TRUE(unifier.unify("{float}", "f32", subst));
}

TEST(UnifierTest, UnifyGenericFloatWithInt) {
  Unifier unifier;
  Substitution subst;
  EXPECT_FALSE(unifier.unify("{float}", "i64", subst));
}

TEST(UnifierTest, UnifyVarWithGenericInt) {
  Unifier unifier;
  Substitution subst;
  EXPECT_TRUE(unifier.unify("$0", "{int}", subst));
  EXPECT_EQ(subst.apply("$0"), "{int}");
}

TEST(UnifierTest, UnifyTypeParamWithConcrete) {
  Unifier unifier;
  Substitution subst;
  // Type parameters ('a, 'b) are not unification vars — they should unify
  // like concrete types
  EXPECT_FALSE(unifier.unify("'a", "i64", subst));
}

TEST(UnifierTest, UnifyTypeParamSame) {
  Unifier unifier;
  Substitution subst;
  EXPECT_TRUE(unifier.unify("'a", "'a", subst));
}

// ============== TraitConstraints Tests ==============

TEST(TraitConstraintsTest, AddAndGetBounds) {
  TraitConstraints tc;
  tc.addBound("$0", TraitBound::Numeric);
  auto bounds = tc.getBounds("$0");
  EXPECT_EQ(bounds.size(), 1u);
  EXPECT_TRUE(bounds.count(TraitBound::Numeric) > 0);
}

TEST(TraitConstraintsTest, MultipleBounds) {
  TraitConstraints tc;
  tc.addBound("$0", TraitBound::Numeric);
  tc.addBound("$0", TraitBound::Integer);
  auto bounds = tc.getBounds("$0");
  EXPECT_EQ(bounds.size(), 2u);
  EXPECT_TRUE(bounds.count(TraitBound::Numeric) > 0);
  EXPECT_TRUE(bounds.count(TraitBound::Integer) > 0);
}

TEST(TraitConstraintsTest, NoBounds) {
  TraitConstraints tc;
  auto bounds = tc.getBounds("$0");
  EXPECT_TRUE(bounds.empty());
}

TEST(TraitConstraintsTest, SatisfiesNumeric) {
  // Numeric is satisfied by integer and float types
  EXPECT_TRUE(
      TraitConstraints::satisfies("i64", {TraitBound::Numeric}));
  EXPECT_TRUE(
      TraitConstraints::satisfies("u32", {TraitBound::Numeric}));
  EXPECT_TRUE(
      TraitConstraints::satisfies("f64", {TraitBound::Numeric}));
  EXPECT_FALSE(
      TraitConstraints::satisfies("bool", {TraitBound::Numeric}));
}

TEST(TraitConstraintsTest, SatisfiesInteger) {
  EXPECT_TRUE(
      TraitConstraints::satisfies("i64", {TraitBound::Integer}));
  EXPECT_TRUE(
      TraitConstraints::satisfies("u8", {TraitBound::Integer}));
  EXPECT_FALSE(
      TraitConstraints::satisfies("f64", {TraitBound::Integer}));
  EXPECT_FALSE(
      TraitConstraints::satisfies("bool", {TraitBound::Integer}));
}

TEST(TraitConstraintsTest, SatisfiesFloat) {
  EXPECT_TRUE(
      TraitConstraints::satisfies("f64", {TraitBound::Float}));
  EXPECT_TRUE(
      TraitConstraints::satisfies("f32", {TraitBound::Float}));
  EXPECT_FALSE(
      TraitConstraints::satisfies("i64", {TraitBound::Float}));
  EXPECT_FALSE(
      TraitConstraints::satisfies("bool", {TraitBound::Float}));
}

TEST(TraitConstraintsTest, SatisfiesMultipleBounds) {
  // Integer implies Numeric, so i64 satisfies both
  EXPECT_TRUE(TraitConstraints::satisfies(
      "i64", {TraitBound::Numeric, TraitBound::Integer}));
  // f64 satisfies Numeric but not Integer
  EXPECT_FALSE(TraitConstraints::satisfies(
      "f64", {TraitBound::Numeric, TraitBound::Integer}));
}

// ============== TraitBound string conversion Tests ==============

TEST(TraitBoundTest, ToString) {
  EXPECT_EQ(traitBoundToString(TraitBound::Numeric), "Numeric");
  EXPECT_EQ(traitBoundToString(TraitBound::Integer), "Integer");
  EXPECT_EQ(traitBoundToString(TraitBound::Float), "Float");
}

TEST(TraitBoundTest, FromString) {
  EXPECT_EQ(stringToTraitBound("Numeric"), TraitBound::Numeric);
  EXPECT_EQ(stringToTraitBound("Integer"), TraitBound::Integer);
  EXPECT_EQ(stringToTraitBound("Float"), TraitBound::Float);
}

TEST(TraitBoundTest, FromStringInvalid) {
  EXPECT_EQ(stringToTraitBound("invalid"), std::nullopt);
  EXPECT_EQ(stringToTraitBound(""), std::nullopt);
}

// ============== Unification Variable Utilities Tests ==============

TEST(UnificationVarTest, IsUnificationVar) {
  EXPECT_TRUE(isUnificationVar("$0"));
  EXPECT_TRUE(isUnificationVar("$1"));
  EXPECT_TRUE(isUnificationVar("$42"));
  EXPECT_FALSE(isUnificationVar("i64"));
  EXPECT_FALSE(isUnificationVar("'a"));
  EXPECT_FALSE(isUnificationVar(""));
}

TEST(UnificationVarTest, FreshUnificationVar) {
  resetUnificationVarCounter();
  EXPECT_EQ(freshUnificationVar(), "$0");
  EXPECT_EQ(freshUnificationVar(), "$1");
  EXPECT_EQ(freshUnificationVar(), "$2");
}

TEST(UnificationVarTest, IsTypeParameter) {
  EXPECT_TRUE(isTypeParameter("'a"));
  EXPECT_TRUE(isTypeParameter("'b"));
  EXPECT_TRUE(isTypeParameter("'abc"));
  EXPECT_FALSE(isTypeParameter("$0"));
  EXPECT_FALSE(isTypeParameter("i64"));
  EXPECT_FALSE(isTypeParameter(""));
}

// ============== TypeScheme Tests ==============

TEST(TypeSchemeTest, MonomorphicScheme) {
  TypeScheme scheme;
  scheme.paramTypes = {"i64", "i64"};
  scheme.returnType = "i64";
  EXPECT_TRUE(scheme.typeParams.empty());
  EXPECT_TRUE(scheme.paramBounds.empty());
}

TEST(TypeSchemeTest, PolymorphicScheme) {
  TypeScheme scheme;
  scheme.typeParams = {"'a"};
  scheme.paramBounds["'a"] = {TraitBound::Numeric};
  scheme.paramTypes = {"'a", "'a"};
  scheme.returnType = "'a";
  EXPECT_EQ(scheme.typeParams.size(), 1u);
  EXPECT_TRUE(scheme.paramBounds.at("'a").count(TraitBound::Numeric) > 0);
}
