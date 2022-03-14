/*
HtmlAnim - A C++ header-only library for creating HTML/JavaScript animations

https://github.com/rkibria/HtmlAnim

MIT License

Copyright (c) 2019 Raihan Kibria

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#pragma once

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace HtmlAnim {

#ifndef M_PI
constexpr double PI = 3.14159265358979323846;
#else
constexpr double PI = M_PI;
#endif

using CoordType = double;

struct Vec2 {
	CoordType x, y;
	explicit Vec2() : x{0}, y{0} {}
	explicit Vec2(CoordType x, CoordType y) : x{x}, y{y} {}

	Vec2 operator-() const {
		return Vec2{-x, -y};
	}

	Vec2& operator+=(const Vec2& rhs) {
		x += rhs.x;
		y += rhs.y;
		return *this;
	}

	Vec2& operator-=(const Vec2& rhs) {
		x -= rhs.x;
		y -= rhs.y;
		return *this;
	}

	Vec2& operator*=(CoordType a) {
		x *= a;
		y *= a;
		return *this;
	}
};

Vec2 operator+(const Vec2& lhs, const Vec2& rhs) {return Vec2{lhs.x + rhs.x, lhs.y + lhs.y};}
Vec2 operator-(const Vec2& lhs, const Vec2& rhs) {return Vec2{lhs.x - rhs.x, lhs.y - lhs.y};}
Vec2 operator*(const CoordType& lhs, const Vec2& rhs) {return Vec2{lhs * rhs.x, lhs * rhs.y};}
Vec2 operator*(const Vec2& lhs, const CoordType& rhs) {return Vec2{lhs.x * rhs, lhs.y * rhs};}

using Vec2Vector = std::vector<Vec2>;

using SizeType = unsigned int;

constexpr SizeType FPS = 60;

std::string rgb_color(SizeType r, SizeType g, SizeType b) {
	std::stringstream ss;
	ss << "#" << std::hex << std::setw(2) << std::setfill('0') << (r % 256)
		<< std::setw(2) << (g % 256) << std::setw(2) << (b % 256);
	return ss.str();
}

using HashType = size_t;
using TypeHashSet = std::unordered_set<HashType>;

class DefinitionsStream {
private:
	std::ostream &output_stream;
	TypeHashSet defined_drawables;

public:
	explicit DefinitionsStream(std::ostream &os) : output_stream{os} {}

	bool is_drawable_defined(const HashType &hash) const {
		return (defined_drawables.find(hash) != defined_drawables.end());
	}

	void set_drawable_defined(const HashType &hash) {
		defined_drawables.insert(hash);
	}

	void write_if_undefined(const HashType &hash, const char* def_code) {
		if(is_drawable_defined(hash))
			return;
		set_drawable_defined(hash);
		output_stream << def_code;
	}

	auto& stream() {return output_stream;}
};

class Drawable {
public:
	virtual ~Drawable() {}

	virtual void define(DefinitionsStream&) const {}
	virtual void draw(std::ostream &os) const = 0;
};

class ExpressionValue {
protected:
	std::string str_val;
public:
	virtual ~ExpressionValue() = default;
	ExpressionValue(const std::string& v) : str_val{ v } {}
	virtual const std::string& to_string() const { return str_val; }
};

class CoordExpressionValue : public ExpressionValue {
public:
	CoordExpressionValue(const std::string& v) : ExpressionValue{ v } {}
	template<typename T> CoordExpressionValue(const T& v) : ExpressionValue{ std::to_string(v) } {}
};

class BoolExpressionValue : public ExpressionValue {
public:
	BoolExpressionValue(const std::string& v) : ExpressionValue{ v } {}
	BoolExpressionValue(const bool& b) : ExpressionValue{ (b ? "true" : "false") } {}
};

class PointExpressionValue : public ExpressionValue {
	std::string str_val_2;
public:
	PointExpressionValue(const std::string& v, const std::string& v2) : ExpressionValue{ v }, str_val_2{ v2 } {}
	virtual const std::string& to_string_2() const { return str_val_2; }
};

class Expression {
public:
	virtual ~Expression() {}

	virtual void init(std::ostream& os) const {}
	virtual void exit(std::ostream& os) const {}

	virtual const ExpressionValue& value() const = 0;
};

class LinearRangeExpression : public Expression {
	CoordType start, stop;
	SizeType steps;
	CoordExpressionValue var_name;
	static SizeType count;
public:
	LinearRangeExpression(CoordType start, CoordType stop, SizeType steps)
		: start{ start }, stop{ stop }, steps{ steps },
		var_name{ std::string("layer.expressions.linear_range_") + std::to_string(count++) } {}
	virtual void init(std::ostream& os) const override {
		os << "if(" << var_name.to_string() << " == null) " << var_name.to_string() << " = " << start << ";\n";
	}
	virtual void exit(std::ostream& os) const override {
		if (start < stop) {
			const auto inc = (stop - start) / steps;
			os << "if(" << var_name.to_string() << " < " << stop << ") {\n"
				<< var_name.to_string() << " += " << inc << ";\n"
				<< "layer.repeat_current_frame = true;\n"
				<< "}\n"
				<< "if (" << var_name.to_string() << " > " << stop << ") {\n"
				<< var_name.to_string() << " = " << stop << ";\n"
				<< "}\n";
		}
		else {
			const auto inc = (start - stop) / steps;
			os << "if(" << var_name.to_string() << " > " << stop << ") {\n"
				<< var_name.to_string() << " -= " << inc << ";\n"
				<< "layer.repeat_current_frame = true;\n"
				<< "}\n"
				<< "if (" << var_name.to_string() << " < " << stop << ") {\n"
				<< var_name.to_string() << " = " << stop << ";\n"
				<< "}\n";
		}
	}
	virtual const ExpressionValue& value() const override { return var_name; }
};
SizeType LinearRangeExpression::count = 0;

class LinearTransformExpression : public Expression {
	LinearRangeExpression linear_range;
	CoordExpressionValue transform_var_name;
	std::string transform;
	static SizeType count;
public:
	LinearTransformExpression(CoordType start, CoordType stop, SizeType steps, const std::string& transform)
		: linear_range(start, stop, steps),
		transform_var_name{ std::string("layer.expressions.linear_transform_") + std::to_string(count++) },
		transform{ transform } {}
	virtual void init(std::ostream& os) const override {
		linear_range.init(os);
		const auto& linear_var = linear_range.value().to_string();
		std::stringstream transform_expression;
		for (const auto& c : transform) {
			if (c == 'X') {
				transform_expression << linear_var;
			}
			else {
				transform_expression << c;
			}
		}
		os << transform_var_name.to_string() << " = " << transform_expression.str() << ";\n";
	}
	virtual void exit(std::ostream& os) const override {
		linear_range.exit(os);
	}
	virtual const ExpressionValue& value() const override { return transform_var_name; }
};
SizeType LinearTransformExpression::count = 0;

class LinearPointExpression : public Expression {
	LinearRangeExpression range_1, range_2;
	PointExpressionValue point;
public:
	LinearPointExpression(const Vec2& start, const Vec2& stop, SizeType steps)
		: range_1( start.x, stop.x, steps ), range_2( start.y, stop.y, steps ),
		point(range_1.value().to_string(), range_2.value().to_string()) {}
	virtual void init(std::ostream& os) const override {
		range_1.init(os);
		range_2.init(os);
	}
	virtual void exit(std::ostream& os) const override {
		range_1.exit(os);
		range_2.exit(os);
	}
	virtual const ExpressionValue& value() const override { return point; }
};

class LinearTransformPointExpression : public Expression {
	LinearTransformExpression range_1, range_2;
	PointExpressionValue point;
public:
	LinearTransformPointExpression(const Vec2& start, const Vec2& stop, SizeType steps,
		const std::string& transform_x, const std::string& transform_y)
		: range_1( start.x, stop.x, steps, transform_x ), range_2( start.y, stop.y, steps, transform_y ),
		point(range_1.value().to_string(), range_2.value().to_string()) {}
	virtual void init(std::ostream& os) const override {
		range_1.init(os);
		range_2.init(os);
	}
	virtual void exit(std::ostream& os) const override {
		range_1.exit(os);
		range_2.exit(os);
	}
	virtual const ExpressionValue& value() const override { return point; }
};

class Arc : public Drawable {
	CoordExpressionValue x, y, r, sa, ea;
	BoolExpressionValue fill;
public:
	explicit Arc(const CoordExpressionValue& x, const CoordExpressionValue& y, const CoordExpressionValue& r,
		const CoordExpressionValue& sa, const CoordExpressionValue& ea, const BoolExpressionValue& fill)
		: x{ x }, y{ y }, r{ r }, sa{ sa }, ea{ ea }, fill{ fill } {}
	virtual void define(DefinitionsStream& ds) const override {
		ds.write_if_undefined(typeid(Arc).hash_code(), R"(
function arc(ctx, x, y, r, sa, ea, fill) {
	ctx.beginPath();
	ctx.arc(x, y, r, sa, ea);
	if(fill)
		ctx.fill();
	else
		ctx.stroke();
}
)");
	}
	virtual void draw(std::ostream &os) const override {
		os << "arc(ctx, " << x.to_string() << ", "
			<< y.to_string() << ", "
			<< r.to_string() << ", "
			<< sa.to_string() << ", "
			<< ea.to_string() << ", "
			<< fill.to_string() << ");\n";
	}
};

class Rect : public Drawable {
	CoordExpressionValue x, y, w, h;
	BoolExpressionValue fill;
public:
	explicit Rect(const CoordExpressionValue& x, const CoordExpressionValue& y,
		const CoordExpressionValue& w, const CoordExpressionValue& h,
		const BoolExpressionValue& fill)
		: x{ x }, y{ y }, w{ w }, h{ h }, fill{ fill } {}
	virtual void define(DefinitionsStream &ds) const override {
		ds.write_if_undefined(typeid(Rect).hash_code(), R"(
function rect(ctx, x, y, w, h, fill) {
	ctx.beginPath();
	ctx.rect(x, y, w, h);
	if(fill)
		ctx.fill();
	else
		ctx.stroke();
}
)");
	}
	virtual void draw(std::ostream& os) const override {
		os << "rect(ctx, " << x.to_string() << ", " << y.to_string() << ", "
			<< w.to_string() << ", " << h.to_string() << ", "
			<< fill.to_string() << ");\n";
	}
};

// TODO allow expressions as input
class Line : public Drawable {
	Vec2Vector points;
	bool fill;
	bool close_path;
public:
	explicit Line(CoordType x1, CoordType y1, CoordType x2, CoordType y2)
		: points{Vec2(x1, y1), Vec2(x2, y2)}, fill{false}, close_path{false} {}
	explicit Line(const Vec2Vector& points, bool fill, bool close_path)
		: points{points}, fill{fill}, close_path{close_path} {
		if(points.size() < 2)
			throw std::runtime_error("Need at least 2 points for line");
	}
	virtual void define(DefinitionsStream &ds) const override {
		ds.write_if_undefined(typeid(Line).hash_code(), R"(
function line(ctx, x1, y1, x2, y2) {
	ctx.beginPath();
	ctx.moveTo(x1, y1);
	ctx.lineTo(x2, y2);
	ctx.stroke();
}
)");
	}
	virtual void draw(std::ostream& os) const override {
		if(points.size() == 2) {
			os << "line(ctx, " << static_cast<int>(points[0].x) << ", " << static_cast<int>(points[0].y)
				<< ", " << static_cast<int>(points[1].x) << ", " << static_cast<int>(points[1].y) << ");\n";
		}
		else {
			os << "ctx.beginPath();\n";
			os << "ctx.moveTo(" << static_cast<int>(points[0].x) << ", " << static_cast<int>(points[0].y) << ");\n";
			for(size_t p_i = 1; p_i < points.size(); ++p_i) {
				os << "ctx.lineTo(" << static_cast<int>(points[p_i].x) << ", " << static_cast<int>(points[p_i].y) << ");\n";
			}
			if(close_path)
				os << "ctx.closePath();\n";
			os << (fill ? "ctx.fill();\n" : "ctx.stroke();\n");
		}
	}
};

class Font : public Drawable {
	std::string font;
public:
	explicit Font(const std::string& font) : font{font} {}
	virtual void draw(std::ostream& os) const override
		{os << "ctx.font = \"" << font << "\";\n";}
};

class FillStyle : public Drawable {
	std::string style;
public:
	explicit FillStyle(const std::string& style) : style{style} {}
	virtual void draw(std::ostream& os) const override
		{os << "ctx.fillStyle = \"" << style << "\";\n";}
};

class FillStyleLinearGradient : public Drawable {
	CoordExpressionValue x0, y0, x1, y1;
	std::string color1, color2;
public:
	explicit FillStyleLinearGradient(const CoordExpressionValue& x0, const CoordExpressionValue& y0,
		const CoordExpressionValue& x1, const CoordExpressionValue& y1,
		const std::string& color1, const std::string& color2)
		: x0{ x0 }, y0{ y0 }, x1{ x1 }, y1{ y1 }, color1{ color1 }, color2{ color2 }
	{}
	virtual void draw(std::ostream& os) const override
	{
		os << "var grd = ctx.createLinearGradient("
			<< x0.to_string() << ", "
			<< y0.to_string() << ", "
			<< x1.to_string() << ", "
			<< y1.to_string() << ");\n";
		os << "grd.addColorStop(0, \"" << color1 << "\");\n";
		os << "grd.addColorStop(1, \"" << color2 << "\");\n";
		os << "ctx.fillStyle = grd;\n";
	}
};

class StrokeStyle : public Drawable {
	std::string style;
public:
	explicit StrokeStyle(const std::string& style) : style{style} {}
	virtual void draw(std::ostream& os) const override
		{os << "ctx.strokeStyle = \"" << style << "\";\n";}
};

class LineCap : public Drawable {
	std::string style;
public:
	explicit LineCap(const std::string& style) : style{style} {}
	virtual void draw(std::ostream& os) const override
		{os << "ctx.lineCap = \"" << style << "\";\n";}
};

class LineWidth : public Drawable {
	CoordExpressionValue width;
public:
	explicit LineWidth(const CoordExpressionValue& width) : width{width} {}
	virtual void draw(std::ostream& os) const override
		{os << "ctx.lineWidth = " << width.to_string() << ";\n";}
};

class Text : public Drawable {
	CoordExpressionValue x, y;
	std::string txt;
	BoolExpressionValue fill;
public:
	explicit Text(const CoordExpressionValue& x, const CoordExpressionValue& y, const char* txt, const BoolExpressionValue& fill)
		: x{ x }, y{ y }, txt{ txt }, fill{ fill } {}
	virtual void define(DefinitionsStream &ds) const override {
		ds.write_if_undefined(typeid(Text).hash_code(), R"(
function text(ctx, x, y, txt, fill) {
	ctx.beginPath();
	if(fill)
		ctx.fillText(txt, x, y);
	else
		ctx.strokeText(txt, x, y);
}
)");
	}
	virtual void draw(std::ostream& os) const override {
		os << "text(ctx, " << x.to_string() << ", " << y.to_string()
			<< ", `" << txt << "`, " << fill.to_string() << ");\n";
	}
};

class Scale : public Drawable {
	CoordExpressionValue x, y;
public:
	explicit Scale(const CoordExpressionValue& x, const CoordExpressionValue& y)
		: x{ x }, y{ y } {}
	virtual void draw(std::ostream& os) const override {
		os << "ctx.scale(" << x.to_string() << ", " << y.to_string() << ");\n";
	}
};

class Rotate : public Drawable {
	CoordExpressionValue rot;
public:
	explicit Rotate(const CoordExpressionValue& rot) : rot{ rot } {}
	virtual void draw(std::ostream& os) const override {
		os << "ctx.rotate(" << rot.to_string() << ");\n";
	}
};

class Translate : public Drawable {
	CoordExpressionValue x, y;
public:
	explicit Translate(const CoordExpressionValue& x, const CoordExpressionValue& y)
		: x{ x }, y{ y } {}
	virtual void draw(std::ostream& os) const override {
		os << "ctx.translate(" << x.to_string() << ", " << y.to_string() << ");\n";
	}
};

class DrawMacro : public Drawable {
	std::string name;
public:
	explicit DrawMacro(const std::string& name) : name{name} {}
	virtual void draw(std::ostream& os) const override {
		os << "macro_" << name << "(ctx);\n";
	}
};

class DrawImage : public Drawable {
	SizeType surface;
	CoordExpressionValue sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight;
public:
	explicit DrawImage(SizeType surface,
		const CoordExpressionValue& sx, const CoordExpressionValue& sy,
		const CoordExpressionValue& sWidth, const CoordExpressionValue& sHeight,
		const CoordExpressionValue& dx, const CoordExpressionValue& dy,
		const CoordExpressionValue& dWidth, const CoordExpressionValue& dHeight)
			: surface{surface},
			sx{sx}, sy{sy}, sWidth{sWidth}, sHeight{sHeight}, dx{dx}, dy{dy}, dWidth{dWidth}, dHeight{dHeight}
			{}
	virtual void draw(std::ostream& os) const override {
		os << "ctx.drawImage(surfaces[" << surface << "],"
			<< sx.to_string() << ","
			<< sy.to_string() << ","
			<< sWidth.to_string() << ","
			<< sHeight.to_string() << ","
			<< dx.to_string() << ","
			<< dy.to_string() << ","
			<< dWidth.to_string() << ","
			<< dHeight.to_string() << ");\n";
	}
};

using DrawableVector = std::vector<std::unique_ptr<Drawable>>;
using ExpressionVector = std::vector<std::unique_ptr<Expression>>;

class Frame : public Drawable {
	DrawableVector dwbl_vec;
	ExpressionVector expr_vec;
public:
	Frame() {}

	Frame& add_drawable(std::unique_ptr<Drawable>&& dwbl) {
		dwbl_vec.emplace_back(std::move(dwbl));
		return *this;
	}

	const CoordExpressionValue& add_coord_expression(std::unique_ptr<Expression>&& expr) {
		expr_vec.emplace_back(std::move(expr));
		return dynamic_cast<const CoordExpressionValue&>(expr_vec.back()->value());
	}

	const PointExpressionValue& add_point_expression(std::unique_ptr<Expression>&& expr) {
		expr_vec.emplace_back(std::move(expr));
		return dynamic_cast<const PointExpressionValue&>(expr_vec.back()->value());
	}

	void clear() {dwbl_vec.clear();}

	void define(DefinitionsStream &ds) const override {
		for(auto& dwbl : dwbl_vec) {
			dwbl->define(ds);
		}
	}

	void draw(std::ostream& os) const override {
		for(auto& expr : expr_vec) {
			expr->init(os);
		}
		for(auto& dwbl : dwbl_vec) {
			dwbl->draw(os);
		}
		for (auto& expr : expr_vec) {
			expr->exit(os);
		}
	}

	// DRAWABLE WRAPPERS
	Frame& arc(const CoordExpressionValue& x, const CoordExpressionValue& y, const CoordExpressionValue& r,
		const BoolExpressionValue& fill = false, const CoordExpressionValue& sa = 0.0, const CoordExpressionValue& ea = 2 * PI)
	{
		return add_drawable(std::make_unique<Arc>(x, y, r, sa, ea, fill));
	}
	Frame& arc(const PointExpressionValue& p, const CoordExpressionValue& r,
		const BoolExpressionValue& fill = false, const CoordExpressionValue& sa = 0.0, const CoordExpressionValue & ea = 2 * PI)
	{
		return add_drawable(std::make_unique<Arc>(p.to_string(), p.to_string_2(), r, sa, ea, fill));
	}
	Frame& draw_macro(const std::string& name) {
		return add_drawable(std::make_unique<DrawMacro>(name));
	}
	Frame& fill_style(const std::string& style)
	{
		return add_drawable(std::make_unique<FillStyle>(style));
	}
	Frame& fill_style_linear_gradient(const CoordExpressionValue& x0, const CoordExpressionValue& y0,
		const CoordExpressionValue& x1, const CoordExpressionValue& y1,
		const std::string& color1, const std::string& color2)
	{
		return add_drawable(std::make_unique<FillStyleLinearGradient>(x0, y0, x1, y1, color1, color2));
	}
	Frame& font(const std::string& font)
	{
		return add_drawable(std::make_unique<Font>(font));
	}
	Frame& line(CoordType x1, CoordType y1, CoordType x2, CoordType y2)
	{
		return add_drawable(std::make_unique<Line>(x1, y1, x2, y2));
	}
	Frame& line(const Vec2Vector& points, bool fill = false, bool close_path = false)
	{
		return add_drawable(std::make_unique<Line>(points, fill, close_path));
	}
	Frame& line_cap(const std::string& style)
	{
		return add_drawable(std::make_unique<LineCap>(style));
	}
	Frame& line_width(const CoordExpressionValue& width)
	{
		return add_drawable(std::make_unique<LineWidth>(width));
	}
	Frame& rect(const CoordExpressionValue& x, const CoordExpressionValue& y, const CoordExpressionValue& w, const CoordExpressionValue& h, const BoolExpressionValue& fill = false)
	{
		return add_drawable(std::make_unique<Rect>(x, y, w, h, fill));
	}
	Frame& rotate(const CoordExpressionValue& rot)
	{
		return add_drawable(std::make_unique<Rotate>(rot));
	}
	Frame& scale(const CoordExpressionValue& x, const CoordExpressionValue& y)
	{
		return add_drawable(std::make_unique<Scale>(x, y));
	}
	Frame& stroke_style(const std::string& style)
	{
		return add_drawable(std::make_unique<StrokeStyle>(style));
	}
	Frame& text(const CoordExpressionValue& x, const CoordExpressionValue& y, std::string txt, const BoolExpressionValue& fill = true)
	{
		return add_drawable(std::make_unique<Text>(x, y, txt.c_str(), fill));
	}
	Frame& translate(const CoordExpressionValue& x, const CoordExpressionValue& y)
	{
		return add_drawable(std::make_unique<Translate>(x, y));
	}
	Frame& wait(SizeType n_frames)
	{
		add_coord_expression(std::make_unique<LinearRangeExpression>(0, n_frames, n_frames));
		return *this;
	}
	Frame& drawImage(SizeType surface, const CoordExpressionValue& sx, const CoordExpressionValue& sy,
		const CoordExpressionValue& sWidth, const CoordExpressionValue& sHeight,
		const CoordExpressionValue& dx, const CoordExpressionValue& dy,
		const CoordExpressionValue& dWidth, const CoordExpressionValue& dHeight)
	{
		return add_drawable(std::make_unique<DrawImage>(surface, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight));
	}

	// EXPRESSION WRAPPERS
	const PointExpressionValue& linear_point_range(const Vec2& start, const Vec2& stop, SizeType steps)
	{
		return add_point_expression(std::make_unique<LinearPointExpression>(start, stop, steps));
	}
	const CoordExpressionValue& linear_range(CoordType start, CoordType stop, SizeType steps)
	{
		return add_coord_expression(std::make_unique<LinearRangeExpression>(start, stop, steps));
	}
	const CoordExpressionValue& linear_transform(CoordType start, CoordType stop, SizeType steps, const std::string& transform)
	{
		return add_coord_expression(std::make_unique<LinearTransformExpression>(start, stop, steps, transform));
	}
	const PointExpressionValue& linear_transform_point(const Vec2& start, const Vec2& stop, SizeType steps,
		const std::string& transform_x, const std::string& transform_y)
	{
		return add_point_expression(std::make_unique<LinearTransformPointExpression>(start, stop, steps,
			transform_x, transform_y));
	}

	// TWEENING EXPRESSIONS
	const CoordExpressionValue& ease_in(CoordType begin, CoordType change, CoordType duration_sec, CoordType strength = 2)
	{
		const auto steps = static_cast<SizeType>(duration_sec * FPS);
		const auto transform = std::to_string(change) + " * Math.pow(X, " + std::to_string(strength) + ") + " + std::to_string(begin);
		return add_coord_expression(std::make_unique<LinearTransformExpression>(0, 1, steps, transform));
	}
	const CoordExpressionValue& ease_out(CoordType begin, CoordType change, CoordType duration_sec, CoordType strength = 2)
	{
		const auto steps = static_cast<SizeType>(duration_sec * FPS);
		const auto transform = std::to_string(change) + " * (1 - Math.pow(1 - X, " + std::to_string(strength) + ")) + " + std::to_string(begin);
		return add_coord_expression(std::make_unique<LinearTransformExpression>(0, 1, steps, transform));
	}
	const CoordExpressionValue& linear_tween(CoordType begin, CoordType change, CoordType duration_sec)
	{
		const auto steps = static_cast<SizeType>(duration_sec * FPS);
		const auto transform = std::to_string(change) + " * X + " + std::to_string(begin);
		return add_coord_expression(std::make_unique<LinearTransformExpression>(0, 1, steps, transform));
	}

	Frame& save();
	Frame& define_macro(const std::string& name);
	Frame& surface(SizeType i);
};

class Save : public Frame {
public:
	explicit Save() {}
	virtual void draw(std::ostream& os) const override {
		os << "ctx.save();\n";
		Frame::draw(os);
		os << "ctx.restore();\n";
	}
};

Frame& Frame::save() {
	add_drawable(std::make_unique<Save>());
	return static_cast<Frame&>(*dwbl_vec.back());
}

class Surface : public Frame {
	SizeType surface_id;
public:
	explicit Surface(SizeType i) : surface_id{ i } {}
	virtual void draw(std::ostream& os) const override {
		os << 
			"context_stack.push(ctx);\n" <<
			"ctx = surfaces[" << surface_id << "].getContext('2d');\n";
		Frame::draw(os);
		os << "ctx = context_stack.pop();\n";
	}
};

Frame& Frame::surface(SizeType i) {
	add_drawable(std::make_unique<Surface>(i));
	return static_cast<Frame&>(*dwbl_vec.back());
}

class DefineMacro : public Frame {
	std::string name;
public:
	explicit DefineMacro(const std::string& name) : name{name} {}
	void define(DefinitionsStream &ds) const override {
		Frame::define(ds);
		ds.stream() << "function macro_" << name << "(ctx) {\n";
		Frame::draw(ds.stream());
		ds.stream() << "}\n";
	}
	virtual void draw(std::ostream& os) const override {}
};

Frame& Frame::define_macro(const std::string& name) {
	add_drawable(std::make_unique<DefineMacro>(name));
	return static_cast<Frame&>(*dwbl_vec.back());
}

using FrameVector = std::vector<std::unique_ptr<Frame>>;

class Layer {
private:
	FrameVector frame_vec;
	size_t cur_frame;
	bool no_clear = false;

public:
	Layer() { clear(); }

	void clear() {
		frame_vec.clear();
		cur_frame = 0;
		frame_vec.emplace_back(std::make_unique<Frame>());
	}

	auto& frame() { return *frame_vec[cur_frame]; }
	auto get_num_frames() const { return frame_vec.size(); }
	void rewind() { cur_frame = 0; }
	auto get_frame_index() const { return cur_frame; }
	void set_no_clear(bool do_clear) { no_clear = do_clear; }

	void next_frame() {
		if (cur_frame == frame_vec.size() - 1) {
			frame_vec.emplace_back(std::make_unique<Frame>());
		}
		++cur_frame;
	}

	void remove_last_frame() {
		if (!frame_vec.empty())
			frame_vec.pop_back();
	}

	void write_frames(std::ostream& os) const {
		os << R"({frame_counter: 0,
no_clear : false,
repeat_current_frame : false,
expressions : {},
)";
		os << "frames: [\n";
		for (size_t frame_i = 0; frame_i < frame_vec.size(); ++frame_i) {
			os << "(function(ctx, layer) {\n";
			frame_vec[frame_i]->draw(os);
			os << "}),\n";
		}
		os << "],\n";
		os << "},\n";
	}

	void write_definitions(DefinitionsStream& ds) const {
		for (const auto& frm : frame_vec) {
			frm->define(ds);
		}
	}

};

using LayerVector = std::vector<std::unique_ptr<Layer>>;

class HtmlAnim {
private:
	std::string title;
	SizeType width;
	SizeType height;

	std::stringstream css_style_stream;
	std::stringstream pre_text_stream;
	std::stringstream post_text_stream;

	const std::string canvas_name = "anim_canvas_1";

	LayerVector layer_vec;
	size_t cur_layer{ 0 };
	size_t num_surfaces{ 0 };

	std::string output_file;

public:
	HtmlAnim() { clear(); }
	explicit HtmlAnim(const char* title = "HtmlAnim",
		SizeType width = 1024, SizeType height = 768)
		: title{ title }, width{ width }, height{ height } {
		clear();
		css_style_stream << "body{background-color:#f2f2f2;color:#000000;font-family:sans-serif;font-size:medium;font-weight:normal;}";
	}

	~HtmlAnim()
	{
		if(!output_file.empty()) {
			write_file(output_file.c_str());
		}
	}

	void set_num_surfaces(size_t n) { num_surfaces = n; }

	void clear() {
		layer_vec.clear();
		cur_layer = 0;
		layer_vec.emplace_back(std::make_unique<Layer>());
	}

	auto& css_style() {return css_style_stream;}
	auto& pre_text() {return pre_text_stream;}
	auto& post_text() {return post_text_stream;}

	auto& layer() { return *layer_vec[cur_layer]; }

	void add_layer() {
		if (cur_layer == layer_vec.size() - 1) {
			layer_vec.emplace_back(std::make_unique<Layer>());
		}
		++cur_layer;
	}

	auto& frame() { return layer().frame(); }
	void next_frame() { layer().next_frame(); }

	void write_stream(std::ostream&) const;
	void write_file(const char*) const;

	auto get_width() const {return width;}
	auto get_height() const {return height;}

	void write_file_on_destruct(const std::string& file) { output_file = file; }

private:
	void write_header(std::ostream& os) const;
	void write_canvas(std::ostream& os) const;
	void write_script(std::ostream& os) const;
	void write_definitions(std::ostream& os) const;
	void write_layers(std::ostream& os) const;
	void write_footer(std::ostream& os) const;
};

void HtmlAnim::write_file(const char* path) const {
	std::ofstream outfile;
	outfile.open(path);
	write_stream(outfile);
	outfile.close();
}

void HtmlAnim::write_stream(std::ostream& os) const {
	write_header(os);
	os << pre_text_stream.str() << "\n";
	write_canvas(os);
	write_script(os);
	os << post_text_stream.str() << "\n";
	write_footer(os);
}

void HtmlAnim::write_header(std::ostream& os) const {
	os << R"(<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="Description" content="Generated with HtmlAnim https://github.com/rkibria/HtmlAnim">
)";

	os << "<title>" << title << "</title>\n";

	os << "</head>\n";
	os << "<style type='text/css'>" << css_style_stream.str() << "</style>\n";
	os << "<body>\n";
}

void HtmlAnim::write_canvas(std::ostream& os) const {
	os << "<canvas id='" << canvas_name 
		<< "' width='" << width
		<< "' height='" << height
		<< "'></canvas>\n";
}

void HtmlAnim::write_script(std::ostream& os) const {
	os << "<script>\n";
	os << "<!--\n";
	os << "var canvas = document.getElementById('" << canvas_name << "');\n";
	os << "var offscreens = [];\n";
	os << "var surfaces = [];\n";
	os << "var context_stack = [];\n";

	os << "for (var i = 0; i < " << num_surfaces << "; ++i) {";
	os << R"(
var cv = document.createElement('canvas');
cv.width = canvas.width;
cv.height = canvas.height;
surfaces.push(cv);
}
)";

	write_definitions(os);
	write_layers(os);

	os << R"(
const num_layers = layers.length;

for(var i = 0; i < num_layers; i++) {
	var cv = document.createElement('canvas');
	cv.width = canvas.width;
	cv.height = canvas.height;
	offscreens.push(cv);
}

function draw_layer(ctx, layer) {
		if(layer.frame_counter == 0 || !layer.no_clear)
			ctx.clearRect(0, 0, canvas.width, canvas.height);
		layer.repeat_current_frame = false;
		(layer.frames[layer.frame_counter])(ctx, layer);
		if(!layer.repeat_current_frame) {
			layer.frame_counter = (layer.frame_counter + 1) % layer.frames.length;
			layer.expressions = {};
		}
}

window.onload = function() {
	(function draw_canvas () {
		for (var i = 0; i < num_layers; i++) {
			var ctx = offscreens[i].getContext('2d');
			var layer = layers[i];
			draw_layer(ctx, layer);
		}
		var ctx = canvas.getContext('2d');
		for (var i = 0; i < num_layers; i++) {
			ctx.drawImage(offscreens[i], 0, 0);
		}
		window.requestAnimationFrame(draw_canvas, canvas);
	}());
}

//-->
</script>
<noscript>JavaScript is required to display this content.</noscript>
)";
}

void HtmlAnim::write_definitions(std::ostream& os) const {
	DefinitionsStream ds(os);
	for(const auto& lyr : layer_vec) {
		lyr->write_definitions(ds);
	}
}

void HtmlAnim::write_layers(std::ostream& os) const {
	os << "layers = [\n";
	for (const auto& lyr : layer_vec) {
		lyr->write_frames(os);
	}
	os << "];\n";
}

void HtmlAnim::write_footer(std::ostream& os) const {
	os << R"(
</body>
</html>
)";
}

} // namespace HtmlAnim
