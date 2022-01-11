import fuzzylite as fl
'''
k=3

sum hop: 1937
avg hop: 3.5476190476190474
max hop: 8
min hop: 1

sum dist: 1751500.0
avg dist: 3207.875457875458
max dist: 6000.0
min dist: 300




k=1

sum hop: 444
avg hop: 2.4395604395604398
max hop: 7
min hop: 1
sum dist: 416800
avg dist: 2290.1098901098903
max dist: 4500
min dist: 300
'''


engine = fl.Engine(
    name="ObstacleAvoidance",
    description=""
)
#ramp start, end 경사 오르기.
engine.input_variables = [
    fl.InputVariable(
        name="Distance",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=2500.000,
        lock_range=False,
        terms=[
            fl.Ramp("S", 1000.000, 500.000),
            fl.Trapezoid("M", 500.000, 1000.000, 1500.000, 2000.000),
            fl.Ramp("L", 1500.000, 2000.000)
        ]
    ),
    fl.InputVariable(
            name="Hop",
            description="",
            enabled=True,
            minimum=1.000,
            maximum=8.000,
            lock_range=False,
            terms=[
                fl.Ramp("S", 4.000, 3.000),
                fl.Trapezoid("M", 3.000, 4.000, 5.000, 6.000),
                fl.Ramp("L", 5.000, 6.000)
            ]
    ),
    fl.InputVariable(
            name="Utilization",
            description="",
            enabled=True,
            minimum=0.000,
            maximum=1.000,
            lock_range=False,
            terms=[
                fl.Ramp("L", 0.200, 0.100),
                fl.Trapezoid("M", 0.100, 0.200, 0.400, 0.500),
                fl.Ramp("H", 0.400, 0.500)
            ]
    )

]
engine.output_variables = [
    fl.OutputVariable(
        name="Priority",
        description="",
        enabled=True,
        minimum=0.000,
        maximum=10.000,
        lock_range=False,
        aggregation=fl.Maximum(),
        defuzzifier=fl.Centroid(100),
        lock_previous=False,
        terms=[
            fl.Ramp("VH", 2.000, 0.000),
            fl.Triangle("H", 1.000, 3.000, 5.000),
            fl.Triangle("M", 3.000, 5.000, 7.000),
            fl.Triangle("L", 5.000, 7.000, 9.000),
            fl.Ramp("VL", 8.000, 10.000)
        ]
    )
]

engine.rule_blocks = [
    fl.RuleBlock(
        name="mamdani",
        description="",
        enabled=True,
        conjunction=fl.Minimum(),
        disjunction=fl.Maximum(),
        implication=fl.Minimum(),
        activation=fl.General(),
        rules=[

            fl.Rule.create("if Distance is S and Hop is S and Utilization is L then Priority is VH", engine),
            fl.Rule.create("if Distance is S and Hop is S and Utilization is M then Priority is VH", engine),
            fl.Rule.create("if Distance is S and Hop is S and Utilization is H then Priority is M", engine),

            fl.Rule.create("if Distance is S and Hop is M and Utilization is L then Priority is VH", engine),
            fl.Rule.create("if Distance is S and Hop is M and Utilization is M then Priority is H", engine),
            fl.Rule.create("if Distance is S and Hop is M and Utilization is H then Priority is H", engine),

            fl.Rule.create("if Distance is S and Hop is L and Utilization is L then Priority is VH", engine),
            fl.Rule.create("if Distance is S and Hop is L and Utilization is M then Priority is H", engine),
            fl.Rule.create("if Distance is S and Hop is L and Utilization is H then Priority is M", engine),


            fl.Rule.create("if Distance is M and Hop is S and Utilization is L then Priority is VH", engine),
            fl.Rule.create("if Distance is M and Hop is S and Utilization is M then Priority is M", engine),
            fl.Rule.create("if Distance is M and Hop is S and Utilization is H then Priority is L", engine),

            fl.Rule.create("if Distance is M and Hop is M and Utilization is L then Priority is H", engine),
            fl.Rule.create("if Distance is M and Hop is M and Utilization is M then Priority is M", engine),
            fl.Rule.create("if Distance is M and Hop is M and Utilization is H then Priority is L", engine),

            fl.Rule.create("if Distance is M and Hop is L and Utilization is L then Priority is H", engine),
            fl.Rule.create("if Distance is M and Hop is L and Utilization is M then Priority is M", engine),
            fl.Rule.create("if Distance is M and Hop is L and Utilization is H then Priority is VL", engine),



            fl.Rule.create("if Distance is L and Hop is S and Utilization is L then Priority is M", engine),
            fl.Rule.create("if Distance is L and Hop is S and Utilization is M then Priority is L", engine),
            fl.Rule.create("if Distance is L and Hop is S and Utilization is H then Priority is VL", engine),

            fl.Rule.create("if Distance is L and Hop is M and Utilization is L then Priority is L", engine),
            fl.Rule.create("if Distance is L and Hop is M and Utilization is M then Priority is L", engine),
            fl.Rule.create("if Distance is L and Hop is M and Utilization is H then Priority is VL", engine),

            fl.Rule.create("if Distance is L and Hop is L and Utilization is L then Priority is L", engine),
            fl.Rule.create("if Distance is L and Hop is L and Utilization is M then Priority is VL", engine),
            fl.Rule.create("if Distance is L and Hop is L and Utilization is H then Priority is VL", engine)
        ]
    )
]

engine.input_variable('Distance').value = 100.00
engine.input_variable('Hop').value = 1.00
engine.input_variable('Utilization').value = 0.01
engine.process()
a = engine.output_variable('Priority').value

print(a)



engine.input_variable('Distance').value = 1000.00
engine.input_variable('Hop').value = 4.00
engine.input_variable('Utilization').value = 0.6
engine.process()
a = engine.output_variable('Priority').value

print(a)

engine.input_variable('Distance').value = 5000.00
engine.input_variable('Hop').value = 11.00
engine.input_variable('Utilization').value = 0.6
engine.process()
a = engine.output_variable('Priority').value

print(a)