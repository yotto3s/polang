#include <gtest/gtest.h>

#include "parser/polang_types.hpp"

using namespace polang;

// ============== TypeNames Constants Tests ==============

TEST(PolangTypesTest, TypeNamesConstants) {
  EXPECT_STREQ(TypeNames::INT, "int");
  EXPECT_STREQ(TypeNames::DOUBLE, "double");
  EXPECT_STREQ(TypeNames::BOOL, "bool");
  EXPECT_STREQ(TypeNames::FUNCTION, "function");
  EXPECT_STREQ(TypeNames::UNKNOWN, "unknown");
}

// ============== typeKindToString Tests ==============

TEST(PolangTypesTest, TypeKindToStringInt) {
  EXPECT_STREQ(typeKindToString(TypeKind::Int), "int");
}

TEST(PolangTypesTest, TypeKindToStringDouble) {
  EXPECT_STREQ(typeKindToString(TypeKind::Double), "double");
}

TEST(PolangTypesTest, TypeKindToStringBool) {
  EXPECT_STREQ(typeKindToString(TypeKind::Bool), "bool");
}

TEST(PolangTypesTest, TypeKindToStringFunction) {
  EXPECT_STREQ(typeKindToString(TypeKind::Function), "function");
}

TEST(PolangTypesTest, TypeKindToStringUnknown) {
  EXPECT_STREQ(typeKindToString(TypeKind::Unknown), "unknown");
}

// ============== parseTypeName Tests ==============

TEST(PolangTypesTest, ParseTypeNameInt) {
  auto result = parseTypeName("int");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, TypeKind::Int);
}

TEST(PolangTypesTest, ParseTypeNameDouble) {
  auto result = parseTypeName("double");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, TypeKind::Double);
}

TEST(PolangTypesTest, ParseTypeNameBool) {
  auto result = parseTypeName("bool");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, TypeKind::Bool);
}

TEST(PolangTypesTest, ParseTypeNameFunction) {
  auto result = parseTypeName("function");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, TypeKind::Function);
}

TEST(PolangTypesTest, ParseTypeNameUnknown) {
  auto result = parseTypeName("unknown");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, TypeKind::Unknown);
}

TEST(PolangTypesTest, ParseTypeNameInvalid) {
  auto result = parseTypeName("invalid_type");
  EXPECT_FALSE(result.has_value());
}

TEST(PolangTypesTest, ParseTypeNameEmpty) {
  auto result = parseTypeName("");
  EXPECT_FALSE(result.has_value());
}

TEST(PolangTypesTest, ParseTypeNameCaseSensitive) {
  // Type names are case-sensitive - "Int" should not match "int"
  auto result = parseTypeName("Int");
  EXPECT_FALSE(result.has_value());
}

// ============== isNumericType Tests ==============

TEST(PolangTypesTest, IsNumericTypeInt) { EXPECT_TRUE(isNumericType("int")); }

TEST(PolangTypesTest, IsNumericTypeDouble) {
  EXPECT_TRUE(isNumericType("double"));
}

TEST(PolangTypesTest, IsNumericTypeBool) {
  EXPECT_FALSE(isNumericType("bool"));
}

TEST(PolangTypesTest, IsNumericTypeFunction) {
  EXPECT_FALSE(isNumericType("function"));
}

TEST(PolangTypesTest, IsNumericTypeUnknown) {
  EXPECT_FALSE(isNumericType("unknown"));
}

// ============== isBooleanType Tests ==============

TEST(PolangTypesTest, IsBooleanTypeBool) { EXPECT_TRUE(isBooleanType("bool")); }

TEST(PolangTypesTest, IsBooleanTypeInt) { EXPECT_FALSE(isBooleanType("int")); }

TEST(PolangTypesTest, IsBooleanTypeDouble) {
  EXPECT_FALSE(isBooleanType("double"));
}

// ============== isUnknownType Tests ==============

TEST(PolangTypesTest, IsUnknownTypeUnknown) {
  EXPECT_TRUE(isUnknownType("unknown"));
}

TEST(PolangTypesTest, IsUnknownTypeInt) { EXPECT_FALSE(isUnknownType("int")); }

TEST(PolangTypesTest, IsUnknownTypeEmpty) { EXPECT_FALSE(isUnknownType("")); }

// ============== Round-trip Tests ==============

TEST(PolangTypesTest, RoundTripAllTypes) {
  // Verify that typeKindToString and parseTypeName are inverse operations
  const TypeKind kinds[] = {TypeKind::Int, TypeKind::Double, TypeKind::Bool,
                            TypeKind::Function, TypeKind::Unknown};

  for (TypeKind kind : kinds) {
    const char* str = typeKindToString(kind);
    auto parsed = parseTypeName(str);
    ASSERT_TRUE(parsed.has_value()) << "Failed to parse: " << str;
    EXPECT_EQ(*parsed, kind) << "Round-trip failed for: " << str;
  }
}
