#include <gtest/gtest.h>

#include "parser/polang_types.hpp"

using namespace polang;

// ============== TypeNames Constants Tests ==============

TEST(PolangTypesTest, TypeNamesConstants) {
  // Signed integers
  EXPECT_STREQ(TypeNames::I8, "i8");
  EXPECT_STREQ(TypeNames::I16, "i16");
  EXPECT_STREQ(TypeNames::I32, "i32");
  EXPECT_STREQ(TypeNames::I64, "i64");
  // Unsigned integers
  EXPECT_STREQ(TypeNames::U8, "u8");
  EXPECT_STREQ(TypeNames::U16, "u16");
  EXPECT_STREQ(TypeNames::U32, "u32");
  EXPECT_STREQ(TypeNames::U64, "u64");
  // Floats
  EXPECT_STREQ(TypeNames::F32, "f32");
  EXPECT_STREQ(TypeNames::F64, "f64");
  // Other
  EXPECT_STREQ(TypeNames::BOOL, "bool");
  EXPECT_STREQ(TypeNames::FUNCTION, "function");
  EXPECT_STREQ(TypeNames::UNKNOWN, "unknown");
}

// ============== typeKindToString Tests ==============

TEST(PolangTypesTest, TypeKindToStringInteger) {
  EXPECT_STREQ(typeKindToString(TypeKind::Integer), "integer");
}

TEST(PolangTypesTest, TypeKindToStringFloat) {
  EXPECT_STREQ(typeKindToString(TypeKind::Float), "float");
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

TEST(PolangTypesTest, ParseTypeNameI64) {
  auto result = parseTypeName("i64");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, TypeKind::Integer);
}

TEST(PolangTypesTest, ParseTypeNameI32) {
  auto result = parseTypeName("i32");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, TypeKind::Integer);
}

TEST(PolangTypesTest, ParseTypeNameU64) {
  auto result = parseTypeName("u64");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, TypeKind::Integer);
}

TEST(PolangTypesTest, ParseTypeNameF64) {
  auto result = parseTypeName("f64");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, TypeKind::Float);
}

TEST(PolangTypesTest, ParseTypeNameF32) {
  auto result = parseTypeName("f32");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, TypeKind::Float);
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
  // Type names are case-sensitive - "I64" should not match "i64"
  auto result = parseTypeName("I64");
  EXPECT_FALSE(result.has_value());
}

// ============== isNumericType Tests ==============

TEST(PolangTypesTest, IsNumericTypeI64) { EXPECT_TRUE(isNumericType("i64")); }

TEST(PolangTypesTest, IsNumericTypeI32) { EXPECT_TRUE(isNumericType("i32")); }

TEST(PolangTypesTest, IsNumericTypeU64) { EXPECT_TRUE(isNumericType("u64")); }

TEST(PolangTypesTest, IsNumericTypeF64) { EXPECT_TRUE(isNumericType("f64")); }

TEST(PolangTypesTest, IsNumericTypeF32) { EXPECT_TRUE(isNumericType("f32")); }

TEST(PolangTypesTest, IsNumericTypeBool) {
  EXPECT_FALSE(isNumericType("bool"));
}

TEST(PolangTypesTest, IsNumericTypeFunction) {
  EXPECT_FALSE(isNumericType("function"));
}

TEST(PolangTypesTest, IsNumericTypeUnknown) {
  EXPECT_FALSE(isNumericType("unknown"));
}

// ============== isIntegerType Tests ==============

TEST(PolangTypesTest, IsIntegerTypeI64) { EXPECT_TRUE(isIntegerType("i64")); }

TEST(PolangTypesTest, IsIntegerTypeI32) { EXPECT_TRUE(isIntegerType("i32")); }

TEST(PolangTypesTest, IsIntegerTypeU64) { EXPECT_TRUE(isIntegerType("u64")); }

TEST(PolangTypesTest, IsIntegerTypeF64) { EXPECT_FALSE(isIntegerType("f64")); }

TEST(PolangTypesTest, IsIntegerTypeBool) {
  EXPECT_FALSE(isIntegerType("bool"));
}

// ============== isSignedIntegerType Tests ==============

TEST(PolangTypesTest, IsSignedIntegerTypeI64) {
  EXPECT_TRUE(isSignedIntegerType("i64"));
}

TEST(PolangTypesTest, IsSignedIntegerTypeU64) {
  EXPECT_FALSE(isSignedIntegerType("u64"));
}

// ============== isUnsignedIntegerType Tests ==============

TEST(PolangTypesTest, IsUnsignedIntegerTypeU64) {
  EXPECT_TRUE(isUnsignedIntegerType("u64"));
}

TEST(PolangTypesTest, IsUnsignedIntegerTypeI64) {
  EXPECT_FALSE(isUnsignedIntegerType("i64"));
}

// ============== isFloatType Tests ==============

TEST(PolangTypesTest, IsFloatTypeF64) { EXPECT_TRUE(isFloatType("f64")); }

TEST(PolangTypesTest, IsFloatTypeF32) { EXPECT_TRUE(isFloatType("f32")); }

TEST(PolangTypesTest, IsFloatTypeI64) { EXPECT_FALSE(isFloatType("i64")); }

// ============== getIntegerWidth Tests ==============

TEST(PolangTypesTest, GetIntegerWidthI8) { EXPECT_EQ(getIntegerWidth("i8"), 8); }

TEST(PolangTypesTest, GetIntegerWidthI16) {
  EXPECT_EQ(getIntegerWidth("i16"), 16);
}

TEST(PolangTypesTest, GetIntegerWidthI32) {
  EXPECT_EQ(getIntegerWidth("i32"), 32);
}

TEST(PolangTypesTest, GetIntegerWidthI64) {
  EXPECT_EQ(getIntegerWidth("i64"), 64);
}

TEST(PolangTypesTest, GetIntegerWidthU64) {
  EXPECT_EQ(getIntegerWidth("u64"), 64);
}

TEST(PolangTypesTest, GetIntegerWidthInvalid) {
  EXPECT_EQ(getIntegerWidth("bool"), 0);
}

// ============== getFloatWidth Tests ==============

TEST(PolangTypesTest, GetFloatWidthF32) { EXPECT_EQ(getFloatWidth("f32"), 32); }

TEST(PolangTypesTest, GetFloatWidthF64) { EXPECT_EQ(getFloatWidth("f64"), 64); }

TEST(PolangTypesTest, GetFloatWidthInvalid) {
  EXPECT_EQ(getFloatWidth("i64"), 0);
}

// ============== isBooleanType Tests ==============

TEST(PolangTypesTest, IsBooleanTypeBool) { EXPECT_TRUE(isBooleanType("bool")); }

TEST(PolangTypesTest, IsBooleanTypeI64) { EXPECT_FALSE(isBooleanType("i64")); }

TEST(PolangTypesTest, IsBooleanTypeF64) { EXPECT_FALSE(isBooleanType("f64")); }

// ============== isUnknownType Tests ==============

TEST(PolangTypesTest, IsUnknownTypeUnknown) {
  EXPECT_TRUE(isUnknownType("unknown"));
}

TEST(PolangTypesTest, IsUnknownTypeI64) { EXPECT_FALSE(isUnknownType("i64")); }

TEST(PolangTypesTest, IsUnknownTypeEmpty) { EXPECT_FALSE(isUnknownType("")); }

// ============== Round-trip Tests ==============

TEST(PolangTypesTest, RoundTripAllTypes) {
  // Verify that typeKindToString and parseTypeName are inverse operations
  const TypeKind kinds[] = {TypeKind::Integer, TypeKind::Float, TypeKind::Bool,
                            TypeKind::Function, TypeKind::Unknown};

  for (TypeKind kind : kinds) {
    const char* str = typeKindToString(kind);
    auto parsed = parseTypeName(str);
    ASSERT_TRUE(parsed.has_value()) << "Failed to parse: " << str;
    EXPECT_EQ(*parsed, kind) << "Round-trip failed for: " << str;
  }
}
