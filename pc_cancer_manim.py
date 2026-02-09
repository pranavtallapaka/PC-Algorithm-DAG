from manim import *

def make_cancer_nodes():
    pollution = Rectangle(width=2.8, height=1.2).shift(UP*3 + LEFT*4)
    smoker = Rectangle(width=2.8, height=1.2).shift(UP*3 + RIGHT*4)
    cancer = Rectangle(width=2.8, height=1.2)
    xray = Rectangle(width=2.8, height=1.2).shift(DOWN*3 + LEFT*4)
    dysp = Rectangle(width=2.8, height=1.2).shift(DOWN*3 + RIGHT*4)

    boxes = {
        "Pollution": pollution,
        "Smoker": smoker,
        "Cancer": cancer,
        "Xray": xray,
        "Dyspnoea": dysp,
    }

    labels = {
        name: Text(name, font_size=32).move_to(box)
        for name, box in boxes.items()
    }

    return boxes, labels

class PCSkeletonExplanation(Scene):
    def construct(self):
        positions = {
            "Pollution": LEFT * 4 + UP * 1,
            "Smoker": LEFT * 4 + DOWN * 1,
            "Cancer": ORIGIN,
            "Xray": RIGHT * 4 + UP * 1,
            "Dyspnoea": RIGHT * 4 + DOWN * 1,
        }

        nodes = {
            name: Circle(radius=0.4, color=BLUE).move_to(pos)
            for name, pos in positions.items()
        }
        labels = {
            name: Text(name, font_size=28).next_to(nodes[name], DOWN)
            for name in nodes
        }
      
        title = Text("PC Algorithm: Skeleton & Edge Orientation", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))

        self.play(
            *[FadeIn(nodes[n]) for n in nodes],
            *[Write(labels[n]) for n in labels]
        )

        self.wait(1)

        skeleton_edges = [
            Line(nodes["Pollution"].get_center(), nodes["Cancer"].get_center()),
            Line(nodes["Smoker"].get_center(), nodes["Cancer"].get_center()),
            Line(nodes["Cancer"].get_center(), nodes["Xray"].get_center()),
            Line(nodes["Cancer"].get_center(), nodes["Dyspnoea"].get_center()),
        ]

        skeleton_text = Text(
            "Step 1: PC learns the skeleton\n(using conditional independence tests)",
            font_size=28
        ).to_edge(DOWN)

        self.play(Write(skeleton_text))
        self.play(*[Create(e) for e in skeleton_edges])
        self.wait(2)

        self.play(FadeOut(skeleton_text))

        ci_title = Text(
            "Example: Conditional Independence Test",
            font_size=30
        ).to_edge(DOWN)
        self.play(Write(ci_title))
        self.wait(1)

        ci_statement = MathTex(
            r"\text{Xray} \;\perp\;\text{Pollution} \mid \text{Cancer}",
            font_size=36
        ).next_to(ci_title, UP)

        self.play(Write(ci_statement))
        self.wait(2)

        ci_explanation = Text(
            "PC tests whether Xray and Pollution\n"
            "are independent after conditioning on Cancer",
            font_size=26,
            line_spacing=1.2
        ).to_edge(DOWN)

        self.play(Transform(ci_title, ci_explanation))
        self.wait(2)

        ci_result = Text(
            "If independent → remove edge\n"
            "If dependent → keep edge",
            font_size=26
        ).to_edge(DOWN)

        self.play(Transform(ci_title, ci_result))
        self.wait(2)

        self.play(
            FadeOut(ci_statement),
            FadeOut(ci_title)
        )

        self.play(FadeOut(skeleton_text))

        orient_text = Text(
            "Step 2: Orient edges where direction is identifiable",
            font_size=28
        ).to_edge(DOWN)
        self.play(Write(orient_text))

        directed_edges = [
            Arrow(
                nodes["Pollution"].get_center(),
                nodes["Cancer"].get_center(),
                buff=0.4,
                color=GREEN
            ),
            Arrow(
                nodes["Smoker"].get_center(),
                nodes["Cancer"].get_center(),
                buff=0.4,
                color=GREEN
            ),
        ]

        self.play(
            Transform(skeleton_edges[0], directed_edges[0]),
            Transform(skeleton_edges[1], directed_edges[1])
        )

        self.wait(2)

        self.play(FadeOut(orient_text))

        ambiguity_text = Text(
            "Edges involving Xray and Dyspnoea remain undirected",
            font_size=28
        ).to_edge(DOWN)
        self.play(Write(ambiguity_text))

        self.play(
            skeleton_edges[2].animate.set_color(YELLOW),
            skeleton_edges[3].animate.set_color(YELLOW)
        )

        self.wait(2)

        explanation = Text(
            "PC cannot orient these edges due to Markov equivalence:\n"
            "multiple DAGs encode the same conditional independencies",
            font_size=26,
            line_spacing=1.2
        ).to_edge(DOWN)

        self.play(Transform(ambiguity_text, explanation))
        self.wait(3)


class PCPlusLikelihoodExplanation(Scene):
    def construct(self):
        positions = {
            "Pollution": LEFT * 4 + UP * 1,
            "Smoker": LEFT * 4 + DOWN * 1,
            "Cancer": ORIGIN,
            "Xray": RIGHT * 4 + UP * 1,
            "Dyspnoea": RIGHT * 4 + DOWN * 1,
        }

        nodes = {
            name: Circle(radius=0.4, color=BLUE).move_to(pos)
            for name, pos in positions.items()
        }
        labels = {
            name: Text(name, font_size=28).next_to(nodes[name], DOWN)
            for name in nodes
        }

        title = Text("PC + Likelihood: Resolving Ambiguous Edges", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))

        self.play(
            *[FadeIn(nodes[n]) for n in nodes],
            *[Write(labels[n]) for n in labels]
        )

        edges = {
            "P_C": Arrow(nodes["Pollution"].get_center(), nodes["Cancer"].get_center(), buff=0.4),
            "S_C": Arrow(nodes["Smoker"].get_center(), nodes["Cancer"].get_center(), buff=0.4),
            "C_X": Line(nodes["Cancer"].get_center(), nodes["Xray"].get_center()),
            "C_D": Line(nodes["Cancer"].get_center(), nodes["Dyspnoea"].get_center()),
        }

        self.play(*[Create(e) for e in edges.values()])

        text = Text(
            "PC produces a CPDAG with ambiguous directions",
            font_size=28
        ).to_edge(DOWN)
        self.play(Write(text))
        self.wait(2)

        new_text = Text(
            "Likelihood scoring compares candidate DAGs\nthat share the same skeleton",
            font_size=28,
            line_spacing=1.2
        ).to_edge(DOWN)
        self.play(Transform(text, new_text))
        self.wait(2)

        arrow_cx = Arrow(
            nodes["Cancer"].get_center(),
            nodes["Xray"].get_center(),
            buff=0.4,
            color=GREEN
        )

        self.play(Transform(edges["C_X"], arrow_cx))

        resolve_text_1 = Text(
            "Cancer → Xray yields higher likelihood\nthan Xray → Cancer",
            font_size=28
        ).to_edge(DOWN)
        self.play(Transform(text, resolve_text_1))
        self.wait(2)

        arrow_cd = Arrow(
            nodes["Cancer"].get_center(),
            nodes["Dyspnoea"].get_center(),
            buff=0.4,
            color=GREEN
        )

        self.play(Transform(edges["C_D"], arrow_cd))

        resolve_text_2 = Text(
            "Cancer → Dyspnoea also maximizes likelihood",
            font_size=28
        ).to_edge(DOWN)
        self.play(Transform(text, resolve_text_2))
        self.wait(2)

        final_text = Text(
            "PC + Likelihood recovers a fully directed DAG",
            font_size=30
        ).to_edge(DOWN)

        self.play(Transform(text, final_text))
        self.wait(3)
