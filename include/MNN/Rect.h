//
//  Rect.h
//  MNN
//
//  Modified by jiangxiaotang on 2018/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

/* Generated by tools/bookmaker from include/core/Rect.h and docs/SkRect_Reference.bmh
   on 2018-07-13 08:15:11. Additional documentation and examples can be found at:
   https://skia.org/user/api/SkRect_Reference

   You may edit either file directly. Structural changes to public interfaces require
   editing both files. After editing docs/SkRect_Reference.bmh, run:
       bookmaker -b docs -i include/core/Rect.h -p
   to create an updated version of this file.
 */

#ifndef SkRect_DEFINED
#define SkRect_DEFINED

#include <math.h>
#include <algorithm>
#include <utility>
#include <MNN/MNNDefine.h>

namespace MNN {
namespace CV {

struct MNN_PUBLIC Point {
    float fX;
    float fY;

    void set(float x, float y);
};

/** \struct Rect
    Rect holds four float coordinates describing the upper and
    lower bounds of a rectangle. Rect may be created from outer bounds or
    from position, width, and height. Rect describes an area; if its right
    is less than or equal to its left, or if its bottom is less than or equal to
    its top, it is considered empty.
*/
struct MNN_PUBLIC Rect {
    float fLeft;   //!< smaller x-axis bounds
    float fTop;    //!< smaller y-axis bounds
    float fRight;  //!< larger x-axis bounds
    float fBottom; //!< larger y-axis bounds

    /** Returns constructed Rect set to (0, 0, 0, 0).
        Many other rectangles are empty; if left is equal to or greater than right,
        or if top is equal to or greater than bottom. Setting all members to zero
        is a convenience, but does not designate a special empty rectangle.

        @return  bounds (0, 0, 0, 0)
    */
    static constexpr Rect MakeEmpty() {
        return Rect{0, 0, 0, 0};
    }

#ifdef SK_SUPPORT_LEGACY_RECTMAKELARGEST
    /** Deprecated.
     */
    static Rect MakeLargest() {
        return {SK_ScalarMin, SK_ScalarMin, SK_ScalarMax, SK_ScalarMax};
    }
#endif

    /** Returns constructed Rect set to float values (0, 0, w, h). Does not
        validate input; w or h may be negative.

        Passing integer values may generate a compiler warning since Rect cannot
        represent 32-bit integers exactly. Use SkIRect for an exact integer rectangle.

        @param w  float width of constructed Rect
        @param h  float height of constructed Rect
        @return   bounds (0, 0, w, h)
    */
    static constexpr Rect MakeWH(float w, float h) {
        return Rect{0, 0, w, h};
    }

    /** Returns constructed Rect set to integer values (0, 0, w, h). Does not validate
        input; w or h may be negative.

        Use to avoid a compiler warning that input may lose precision when stored.
        Use SkIRect for an exact integer rectangle.

        @param w  integer width of constructed Rect
        @param h  integer height of constructed Rect
        @return   bounds (0, 0, w, h)
    */
    static constexpr Rect MakeIWH(int w, int h) {
        return Rect{0, 0, (float)(w), (float)(h)};
    }

    /** Returns constructed Rect set to (l, t, r, b). Does not sort input; Rect may
        result in fLeft greater than fRight, or fTop greater than fBottom.

        @param l  float stored in fLeft
        @param t  float stored in fTop
        @param r  float stored in fRight
        @param b  float stored in fBottom
        @return   bounds (l, t, r, b)
    */
    static constexpr Rect MakeLTRB(float l, float t, float r, float b) {
        return Rect{l, t, r, b};
    }

    /** Returns constructed Rect set to (x, y, x + w, y + h). Does not validate input;
        w or h may be negative.

        @param x  stored in fLeft
        @param y  stored in fTop
        @param w  added to x and stored in fRight
        @param h  added to y and stored in fBottom
        @return   bounds at (x, y) with width w and height h
    */
    static constexpr Rect MakeXYWH(float x, float y, float w, float h) {
        return Rect{x, y, x + w, y + h};
    }

    /** Returns true if fLeft is equal to or greater than fRight, or if fTop is equal
        to or greater than fBottom. Call sort() to reverse rectangles with negative
        width() or height().

        @return  true if width() or height() are zero or negative
    */
    bool isEmpty() const;

    /** Returns true if fLeft is equal to or less than fRight, or if fTop is equal
        to or less than fBottom. Call sort() to reverse rectangles with negative
        width() or height().

        @return  true if width() or height() are zero or positive
    */
    bool isSorted() const;

    /** Returns left edge of Rect, if sorted. Call isSorted() to see if Rect is valid.
        Call sort() to reverse fLeft and fRight if needed.

        @return  fLeft
    */
    constexpr float x() const {
        return fLeft;
    }

    /** Returns top edge of Rect, if sorted. Call isEmpty() to see if Rect may be invalid,
        and sort() to reverse fTop and fBottom if needed.

        @return  fTop
    */
    constexpr float y() const {
        return fTop;
    }

    /** Returns left edge of Rect, if sorted. Call isSorted() to see if Rect is valid.
        Call sort() to reverse fLeft and fRight if needed.

        @return  fLeft
    */
    constexpr float left() const {
        return fLeft;
    }

    /** Returns top edge of Rect, if sorted. Call isEmpty() to see if Rect may be invalid,
        and sort() to reverse fTop and fBottom if needed.

        @return  fTop
    */
    constexpr float top() const {
        return fTop;
    }

    /** Returns right edge of Rect, if sorted. Call isSorted() to see if Rect is valid.
        Call sort() to reverse fLeft and fRight if needed.

        @return  fRight
    */
    constexpr float right() const {
        return fRight;
    }

    /** Returns bottom edge of Rect, if sorted. Call isEmpty() to see if Rect may be invalid,
        and sort() to reverse fTop and fBottom if needed.

        @return  fBottom
    */
    constexpr float bottom() const {
        return fBottom;
    }

    /** Returns span on the x-axis. This does not check if Rect is sorted, or if
        result fits in 32-bit float; result may be negative or infinity.

        @return  fRight minus fLeft
    */
    constexpr float width() const {
        return fRight - fLeft;
    }

    /** Returns span on the y-axis. This does not check if Rect is sorted, or if
        result fits in 32-bit float; result may be negative or infinity.

        @return  fBottom minus fTop
    */
    constexpr float height() const {
        return fBottom - fTop;
    }

    /** Returns average of left edge and right edge. Result does not change if Rect
        is sorted. Result may overflow to infinity if Rect is far from the origin.

        @return  midpoint in x
    */
    constexpr float centerX() const {
        // don't use floatHalf(fLeft + fBottom) as that might overflow before the 0.5
        return 0.5f * (fLeft) + 0.5f * (fRight);
    }

    /** Returns average of top edge and bottom edge. Result does not change if Rect
        is sorted.

        @return  midpoint in y
    */
    constexpr float centerY() const {
        // don't use floatHalf(fTop + fBottom) as that might overflow before the 0.5
        return 0.5f * (fTop) + 0.5f * (fBottom);
    }

    /** Sets Rect to (0, 0, 0, 0).

        Many other rectangles are empty; if left is equal to or greater than right,
        or if top is equal to or greater than bottom. Setting all members to zero
        is a convenience, but does not designate a special empty rectangle.
    */
    void setEmpty();

    /** Sets Rect to (left, top, right, bottom).
        left and right are not sorted; left is not necessarily less than right.
        top and bottom are not sorted; top is not necessarily less than bottom.

        @param left    stored in fLeft
        @param top     stored in fTop
        @param right   stored in fRight
        @param bottom  stored in fBottom
    */
    void set(float left, float top, float right, float bottom);

    /** Sets Rect to (left, top, right, bottom).
        left and right are not sorted; left is not necessarily less than right.
        top and bottom are not sorted; top is not necessarily less than bottom.

        @param left    stored in fLeft
        @param top     stored in fTop
        @param right   stored in fRight
        @param bottom  stored in fBottom
    */
    void setLTRB(float left, float top, float right, float bottom);

    /** Sets Rect to (left, top, right, bottom).
        All parameters are promoted from integer to scalar.
        left and right are not sorted; left is not necessarily less than right.
        top and bottom are not sorted; top is not necessarily less than bottom.

        @param left    promoted to float and stored in fLeft
        @param top     promoted to float and stored in fTop
        @param right   promoted to float and stored in fRight
        @param bottom  promoted to float and stored in fBottom
    */
    void iset(int left, int top, int right, int bottom);

    /** Sets Rect to (0, 0, width, height).
        width and height may be zero or negative. width and height are promoted from
        integer to float, large values may lose precision.

        @param width   promoted to float and stored in fRight
        @param height  promoted to float and stored in fBottom
    */
    void isetWH(int width, int height);

    /** Sets Rect to (x, y, x + width, y + height). Does not validate input;
        width or height may be negative.

        @param x       stored in fLeft
        @param y       stored in fTop
        @param width   added to x and stored in fRight
        @param height  added to y and stored in fBottom
    */
    void setXYWH(float x, float y, float width, float height);

    /** Sets Rect to (0, 0, width, height). Does not validate input;
        width or height may be negative.

        @param width   stored in fRight
        @param height  stored in fBottom
    */
    void setWH(float width, float height);

    /** Returns Rect offset by (dx, dy).

        If dx is negative, Rect returned is moved to the left.
        If dx is positive, Rect returned is moved to the right.
        If dy is negative, Rect returned is moved upward.
        If dy is positive, Rect returned is moved downward.

        @param dx  added to fLeft and fRight
        @param dy  added to fTop and fBottom
        @return    Rect offset on axes, with original width and height
    */
    Rect makeOffset(float dx, float dy) const;

    /** Returns Rect, inset by (dx, dy).

        If dx is negative, Rect returned is wider.
        If dx is positive, Rect returned is narrower.
        If dy is negative, Rect returned is taller.
        If dy is positive, Rect returned is shorter.

        @param dx  added to fLeft and subtracted from fRight
        @param dy  added to fTop and subtracted from fBottom
        @return    Rect inset symmetrically left and right, top and bottom
    */
    Rect makeInset(float dx, float dy) const;

    /** Returns Rect, outset by (dx, dy).

        If dx is negative, Rect returned is narrower.
        If dx is positive, Rect returned is wider.
        If dy is negative, Rect returned is shorter.
        If dy is positive, Rect returned is taller.

        @param dx  subtracted to fLeft and added from fRight
        @param dy  subtracted to fTop and added from fBottom
        @return    Rect outset symmetrically left and right, top and bottom
    */
    Rect makeOutset(float dx, float dy) const;

    /** Offsets Rect by adding dx to fLeft, fRight; and by adding dy to fTop, fBottom.

        If dx is negative, moves Rect to the left.
        If dx is positive, moves Rect to the right.
        If dy is negative, moves Rect upward.
        If dy is positive, moves Rect downward.

        @param dx  offset added to fLeft and fRight
        @param dy  offset added to fTop and fBottom
    */
    void offset(float dx, float dy);

    /** Offsets Rect so that fLeft equals newX, and fTop equals newY. width and height
        are unchanged.

        @param newX  stored in fLeft, preserving width()
        @param newY  stored in fTop, preserving height()
    */
    void offsetTo(float newX, float newY);

    /** Insets Rect by (dx, dy).

        If dx is positive, makes Rect narrower.
        If dx is negative, makes Rect wider.
        If dy is positive, makes Rect shorter.
        If dy is negative, makes Rect taller.

        @param dx  added to fLeft and subtracted from fRight
        @param dy  added to fTop and subtracted from fBottom
    */
    void inset(float dx, float dy);

    /** Outsets Rect by (dx, dy).

        If dx is positive, makes Rect wider.
        If dx is negative, makes Rect narrower.
        If dy is positive, makes Rect taller.
        If dy is negative, makes Rect shorter.

        @param dx  subtracted to fLeft and added from fRight
        @param dy  subtracted to fTop and added from fBottom
    */
    void outset(float dx, float dy);

private:
    static bool Intersects(float al, float at, float ar, float ab, float bl, float bt, float br, float bb);

public:
    /** Constructs Rect to intersect from (left, top, right, bottom). Does not sort
        construction.

        Returns true if Rect intersects construction.
        Returns false if either construction or Rect is empty, or do not intersect.

        @param left    x-axis minimum of constructed Rect
        @param top     y-axis minimum of constructed Rect
        @param right   x-axis maximum of constructed Rect
        @param bottom  y-axis maximum of constructed Rect
        @return        true if construction and Rect have area in common
    */
    bool intersects(float left, float top, float right, float bottom) const;

    /** Returns true if Rect intersects r.
        Returns false if either r or Rect is empty, or do not intersect.

        @param r  Rect to intersect
        @return   true if r and Rect have area in common
    */
    bool intersects(const Rect& r) const;

    /** Returns true if a intersects b.
        Returns false if either a or b is empty, or do not intersect.

        @param a  Rect to intersect
        @param b  Rect to intersect
        @return   true if a and b have area in common
    */
    static bool Intersects(const Rect& a, const Rect& b);

    /** Sets Rect to the union of itself and r.

        Asserts if r is empty and SK_DEBUG is defined.
        If Rect is empty, sets Rect to r.

        May produce incorrect results if r is empty.

        @param r  expansion Rect
    */
    void joinNonEmptyArg(const Rect& r);

    /** Sets Rect to the union of itself and the construction.

        May produce incorrect results if Rect or r is empty.

        @param r  expansion Rect
    */
    void joinPossiblyEmptyRect(const Rect& r);

    /** Returns true if: fLeft <= x < fRight && fTop <= y < fBottom.
        Returns false if Rect is empty.

        @param x  test Point x-coordinate
        @param y  test Point y-coordinate
        @return   true if (x, y) is inside Rect
    */
    bool contains(float x, float y) const;

    /** Swaps fLeft and fRight if fLeft is greater than fRight; and swaps
        fTop and fBottom if fTop is greater than fBottom. Result may be empty;
        and width() and height() will be zero or positive.
    */
    void sort();

    /** Returns Rect with fLeft and fRight swapped if fLeft is greater than fRight; and
        with fTop and fBottom swapped if fTop is greater than fBottom. Result may be empty;
        and width() and height() will be zero or positive.

        @return  sorted Rect
    */
    Rect makeSorted() const;

    /** Returns pointer to first scalar in Rect, to treat it as an array with four
        entries.

        @return  pointer to fLeft
    */
    const float* asScalars() const;
};

} // namespace CV
} // namespace MNN
#endif
