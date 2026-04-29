#!/usr/bin/env python3
"""
Synthetic resume dataset generator.

Creates a mixed-quality dataset of resumes with:
- realistic multi-section text
- extracted structured fields
- engineered numerical features
- quality scores and classification labels
- LLM-style feedback (strengths, weaknesses, suggestions)
- balanced domain coverage with optional multi-domain profiles

Outputs:
- CSV file
- JSON file
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DOMAINS: dict[str, dict[str, list[str]]] = {
    "Software Engineering": {
        "roles": [
            "Software Engineer",
            "Backend Developer",
            "Frontend Developer",
            "Full Stack Engineer",
            "DevOps Engineer",
            "QA Automation Engineer",
        ],
        "skills": [
            "Microservices Architecture",
            "API Design",
            "Concurrency",
            "Object-Oriented Design",
            "Test Automation",
            "Performance Tuning",
            "Version Control Workflows",
            "Code Review Practices",
            "Container Orchestration",
            "Event-Driven Systems",
        ],
        "tools": [
            "Python",
            "Java",
            "JavaScript",
            "TypeScript",
            "Go",
            "Node.js",
            "React",
            "Docker",
            "Kubernetes",
            "GitHub Actions",
            "PostgreSQL",
            "Redis",
        ],
        "projects": [
            "Built a service mesh migration plan that reduced deployment incidents by 29%.",
            "Implemented asynchronous job processing to cut API response failures during peak traffic.",
            "Developed a feature flag framework that enabled controlled rollouts across 6 product teams.",
            "Created observability dashboards that improved mean time to recovery by 34%.",
            "Designed a secure authentication gateway supporting SSO and role-based access.",
            "Refactored monolithic modules into maintainable domain services with contract tests.",
        ],
        "experience_templates": [
            "Improved release stability by {metric}% through better automated testing and canary deployments.",
            "Reduced backend latency by {metric}% via caching, query optimization, and async processing.",
            "Led migration of {count} legacy services to containerized infrastructure with zero downtime cutover.",
            "Introduced coding standards and peer review workflow that lowered defect leakage by {metric}%.",
            "Optimized CI pipelines and reduced average build time from {from_time} to {to_time} minutes.",
        ],
        "courses": [
            "Distributed Systems Engineering",
            "Advanced Database Systems",
            "Software Reliability",
            "Cloud Native Architecture",
            "Secure API Development",
            "Scalable System Design",
        ],
    },
    "Data Science": {
        "roles": [
            "Data Scientist",
            "Machine Learning Engineer",
            "Applied Scientist",
            "Analytics Engineer",
            "NLP Engineer",
        ],
        "skills": [
            "Feature Engineering",
            "Statistical Inference",
            "Model Evaluation",
            "Experiment Design",
            "Time Series Forecasting",
            "Predictive Modeling",
            "Data Wrangling",
            "Causal Analysis",
            "Model Interpretability",
            "Anomaly Detection",
        ],
        "tools": [
            "Python",
            "R",
            "SQL",
            "Pandas",
            "NumPy",
            "scikit-learn",
            "PyTorch",
            "TensorFlow",
            "XGBoost",
            "MLflow",
            "Airflow",
            "Tableau",
        ],
        "projects": [
            "Built a demand forecasting system that improved planning accuracy by 18%.",
            "Developed an NLP classification pipeline for support tickets with macro-F1 above 0.90.",
            "Created a churn model and intervention dashboard that reduced monthly churn by 11%.",
            "Designed an anomaly detection service to flag suspicious behavior in near real-time.",
            "Automated feature lineage tracking to improve reproducibility across model versions.",
            "Implemented offline/online metric parity checks before model deployment.",
        ],
        "experience_templates": [
            "Improved model ROC-AUC by {metric} points through feature engineering and hyperparameter tuning.",
            "Reduced forecast error by {metric}% after redesigning time-series feature pipelines.",
            "Built experiment monitoring that increased trust in A/B test readouts across {count} teams.",
            "Delivered automated analytics reports that saved {count} analyst hours per month.",
            "Productionized ML workflows and reduced retraining cycle time by {metric}%.",
        ],
        "courses": [
            "Applied Machine Learning",
            "Deep Learning Systems",
            "Applied Statistics",
            "Natural Language Processing",
            "Bayesian Methods",
            "MLOps Fundamentals",
        ],
    },
    "HR": {
        "roles": [
            "HR Generalist",
            "Talent Acquisition Specialist",
            "People Operations Partner",
            "Recruitment Coordinator",
            "Learning and Development Associate",
        ],
        "skills": [
            "Candidate Sourcing",
            "Onboarding Design",
            "Interview Calibration",
            "Employee Relations",
            "Policy Interpretation",
            "Compensation Benchmarking",
            "Succession Planning",
            "Workforce Planning",
            "Conflict Mediation",
            "Performance Review Facilitation",
        ],
        "tools": [
            "Workday",
            "BambooHR",
            "Greenhouse",
            "Lever",
            "Excel",
            "Power BI",
            "CultureAmp",
            "SAP SuccessFactors",
            "Google Workspace",
            "Notion",
        ],
        "projects": [
            "Designed a structured interview framework that increased hiring consistency across departments.",
            "Launched a manager onboarding program that improved first-90-day retention metrics.",
            "Built headcount planning dashboards for quarterly workforce forecasting.",
            "Introduced candidate experience scorecards and improved acceptance rates.",
            "Reworked policy communication templates to reduce compliance escalations.",
            "Implemented an internal mobility campaign boosting role transition visibility.",
        ],
        "experience_templates": [
            "Reduced average time-to-fill by {metric}% by optimizing sourcing channels and interview scheduling.",
            "Improved employee engagement survey participation by {metric}% through targeted communication.",
            "Built workforce reports used by leadership to plan hiring for {count} business units.",
            "Led policy refresh cycle and improved manager policy adherence by {metric}%.",
            "Created onboarding process maps that shortened new-hire ramp-up by {metric}%.",
        ],
        "courses": [
            "Strategic Human Resource Management",
            "Compensation and Benefits",
            "Organizational Development",
            "Employment Law Basics",
            "People Analytics",
            "Learning Program Design",
        ],
    },
    "Marketing": {
        "roles": [
            "Digital Marketing Specialist",
            "Growth Marketer",
            "SEO Strategist",
            "Campaign Manager",
            "Content Marketing Lead",
        ],
        "skills": [
            "Funnel Optimization",
            "Audience Segmentation",
            "Campaign Planning",
            "Brand Messaging",
            "Conversion Copywriting",
            "Lifecycle Marketing",
            "Attribution Analysis",
            "Creative Briefing",
            "Landing Page Testing",
            "Retention Strategy",
        ],
        "tools": [
            "Google Analytics 4",
            "Google Ads",
            "Meta Ads Manager",
            "HubSpot",
            "Mailchimp",
            "Semrush",
            "Ahrefs",
            "Canva",
            "Figma",
            "Looker Studio",
        ],
        "projects": [
            "Scaled qualified leads by 31% through multi-channel campaign orchestration.",
            "Built conversion-focused content hubs that increased non-branded traffic significantly.",
            "Designed lifecycle email automations that improved repeat conversion rates.",
            "Created attribution dashboards to align media spend with pipeline outcomes.",
            "Ran landing page experiment program and improved average conversion by 14%.",
            "Refined audience personas and messaging for two major product launches.",
        ],
        "experience_templates": [
            "Improved paid campaign ROAS by {metric}% by refining audience and bid strategies.",
            "Increased organic sessions by {metric}% with technical SEO and content cluster planning.",
            "Raised email click-through rate by {metric}% using segmentation and personalization.",
            "Built weekly performance reporting for {count} channels and marketing leadership.",
            "Reduced customer acquisition cost by {metric}% through creative and landing page optimization.",
        ],
        "courses": [
            "Performance Marketing",
            "SEO Strategy",
            "Brand Management",
            "Marketing Analytics",
            "Content Strategy",
            "Consumer Behavior",
        ],
    },
    "Finance": {
        "roles": [
            "Financial Analyst",
            "Investment Analyst",
            "Corporate Finance Associate",
            "Risk Analyst",
            "FP&A Analyst",
        ],
        "skills": [
            "Financial Modeling",
            "Valuation Analysis",
            "Budget Forecasting",
            "Cash Flow Planning",
            "Risk Assessment",
            "Variance Analysis",
            "Scenario Planning",
            "Portfolio Analysis",
            "Cost Optimization",
            "Regulatory Reporting",
        ],
        "tools": [
            "Advanced Excel",
            "PowerPoint",
            "Bloomberg Terminal",
            "SQL",
            "Python",
            "Tableau",
            "Power BI",
            "QuickBooks",
            "SAP FICO",
            "Oracle Financials",
        ],
        "projects": [
            "Built rolling forecast models improving quarterly forecast accuracy by 16%.",
            "Developed working capital dashboards for treasury and operations leaders.",
            "Automated variance analysis reports to accelerate monthly close insights.",
            "Created scenario models to evaluate pricing and margin sensitivity.",
            "Designed risk monitoring scorecards for exposure tracking across business units.",
            "Standardized KPI definitions across FP&A and controllership teams.",
        ],
        "experience_templates": [
            "Reduced budget variance by {metric}% through tighter forecasting assumptions and controls.",
            "Built valuation models for {count} opportunities supporting investment decision processes.",
            "Improved monthly close reporting turnaround by {metric}% via automation.",
            "Implemented risk control checks that reduced reporting discrepancies by {metric}%.",
            "Supported executive planning cycles with scenario analysis for revenue and cost outlook.",
        ],
        "courses": [
            "Corporate Finance",
            "Financial Statement Analysis",
            "Investment Analysis",
            "Risk Management",
            "Managerial Accounting",
            "Financial Markets",
        ],
    },
    "Sales": {
        "roles": [
            "Sales Executive",
            "Account Executive",
            "Business Development Representative",
            "Enterprise Sales Manager",
            "Inside Sales Specialist",
        ],
        "skills": [
            "Lead Qualification",
            "Consultative Selling",
            "Negotiation Strategy",
            "Pipeline Management",
            "Account Mapping",
            "Territory Planning",
            "Objection Handling",
            "Deal Forecasting",
            "Client Retention",
            "Proposal Development",
        ],
        "tools": [
            "Salesforce",
            "HubSpot CRM",
            "LinkedIn Sales Navigator",
            "Outreach",
            "Apollo",
            "Gong",
            "ZoomInfo",
            "Excel",
            "Power BI",
            "DocuSign",
        ],
        "projects": [
            "Built a territory expansion playbook that improved new logo conversions.",
            "Redesigned pipeline hygiene process to improve forecast reliability.",
            "Launched outbound sequence experiments for strategic prospect segments.",
            "Created account intelligence templates for complex enterprise pursuits.",
            "Implemented renewal risk tracker with customer success collaboration.",
            "Standardized proposal workflow reducing quote-to-close cycle time.",
        ],
        "experience_templates": [
            "Exceeded annual quota by {metric}% through focused account planning and multithreading.",
            "Improved win rate by {metric}% by refining discovery and qualification criteria.",
            "Increased pipeline coverage to {count}x target through outbound and partner channels.",
            "Raised renewal rates by {metric}% through proactive engagement and risk tracking.",
            "Reduced average sales cycle from {from_time} to {to_time} days for mid-market deals.",
        ],
        "courses": [
            "Strategic Sales Management",
            "Enterprise Selling",
            "Negotiation and Influence",
            "Revenue Operations",
            "Customer Success Foundations",
            "B2B Prospecting Techniques",
        ],
    },
    "Product Management": {
        "roles": [
            "Product Manager",
            "Associate Product Manager",
            "Technical Product Manager",
            "Growth Product Manager",
            "Platform Product Manager",
        ],
        "skills": [
            "Product Discovery",
            "Roadmap Prioritization",
            "User Story Writing",
            "KPI Definition",
            "Stakeholder Alignment",
            "Backlog Management",
            "Go-to-Market Planning",
            "Hypothesis Testing",
            "Product Analytics",
            "Outcome-Driven Planning",
        ],
        "tools": [
            "Jira",
            "Confluence",
            "Amplitude",
            "Mixpanel",
            "Looker",
            "Figma",
            "Miro",
            "Notion",
            "Aha!",
            "SQL",
        ],
        "projects": [
            "Defined and launched onboarding improvements increasing activation by 13%.",
            "Built prioritization framework tying roadmap bets to measurable outcomes.",
            "Led discovery interviews and prototyping for a high-impact workflow redesign.",
            "Introduced experiment review rituals to strengthen decision quality.",
            "Reduced support ticket volume by simplifying key product workflows.",
            "Coordinated release readiness process across engineering, support, and sales.",
        ],
        "experience_templates": [
            "Improved feature adoption by {metric}% by aligning roadmap priorities to user pain points.",
            "Reduced time-to-value by {metric}% through onboarding and activation experiments.",
            "Managed cross-functional delivery for {count} roadmap initiatives per quarter.",
            "Created product KPI scorecards used in weekly executive reviews.",
            "Improved release predictability by introducing milestone-based planning discipline.",
        ],
        "courses": [
            "Product Strategy",
            "Agile Product Development",
            "Product Analytics",
            "Design Thinking for Product",
            "Go-to-Market Execution",
            "Behavioral Economics for PMs",
        ],
    },
    "UI/UX Design": {
        "roles": [
            "UI Designer",
            "UX Designer",
            "Product Designer",
            "Interaction Designer",
            "UX Researcher",
        ],
        "skills": [
            "User Research",
            "Wireframing",
            "Interaction Design",
            "Information Architecture",
            "Accessibility Design",
            "Journey Mapping",
            "Visual Hierarchy",
            "Usability Testing",
            "Design Systems",
            "Prototyping",
        ],
        "tools": [
            "Figma",
            "Adobe XD",
            "Sketch",
            "Miro",
            "InVision",
            "FigJam",
            "Maze",
            "Hotjar",
            "Zeplin",
            "Framer",
        ],
        "projects": [
            "Redesigned checkout flow and improved completion rate by 17%.",
            "Built reusable design components that reduced handoff inconsistencies.",
            "Conducted usability studies and resolved major friction points in onboarding.",
            "Created responsive design guidelines adopted across mobile and web surfaces.",
            "Developed research repository for continuous discovery and insight tracking.",
            "Collaborated with engineering to improve accessibility compliance in core workflows.",
        ],
        "experience_templates": [
            "Improved usability test success rate by {metric}% after iterative prototype refinements.",
            "Reduced design-to-development handoff cycles by {metric}% through design system updates.",
            "Ran {count} moderated interviews to validate navigation and content architecture.",
            "Increased mobile task completion by {metric}% by simplifying interaction patterns.",
            "Established accessibility checklists used in weekly design reviews.",
        ],
        "courses": [
            "Human-Computer Interaction",
            "User Experience Research",
            "Visual Interface Design",
            "Accessibility in Product Design",
            "Design Systems Practice",
            "Service Design Basics",
        ],
    },
    "Cybersecurity": {
        "roles": [
            "Security Analyst",
            "SOC Analyst",
            "Cybersecurity Engineer",
            "Incident Response Specialist",
            "Security Operations Engineer",
        ],
        "skills": [
            "Threat Hunting",
            "Vulnerability Assessment",
            "Incident Triage",
            "Network Security Monitoring",
            "Identity and Access Management",
            "Security Hardening",
            "Log Analysis",
            "Cloud Security Posture",
            "Penetration Testing Basics",
            "Risk Mitigation Planning",
        ],
        "tools": [
            "Splunk",
            "Wireshark",
            "Nessus",
            "Burp Suite",
            "CrowdStrike",
            "Microsoft Sentinel",
            "Kali Linux",
            "SIEM",
            "SOAR",
            "OpenVAS",
        ],
        "projects": [
            "Built a phishing incident playbook reducing triage time during alerts.",
            "Implemented SIEM detection rules that increased true positive rates.",
            "Automated log correlation workflows for suspicious authentication patterns.",
            "Conducted vulnerability remediation campaign across critical internal assets.",
            "Designed role-based access review process with security and IT teams.",
            "Created security awareness simulations and tracked risk reduction outcomes.",
        ],
        "experience_templates": [
            "Reduced incident response time by {metric}% through playbook and automation improvements.",
            "Identified and remediated {count} high-severity vulnerabilities before exploitation.",
            "Improved alert precision by tuning SIEM rules and reducing false positives by {metric}%.",
            "Led post-incident reviews and implemented preventive controls for recurring threats.",
            "Hardened endpoint and access policies, lowering policy violations by {metric}%.",
        ],
        "courses": [
            "Network Security",
            "Ethical Hacking Fundamentals",
            "Security Operations and Monitoring",
            "Cloud Security Essentials",
            "Digital Forensics Basics",
            "Cyber Risk Management",
        ],
    },
    "Business Analyst": {
        "roles": [
            "Business Analyst",
            "Senior Business Analyst",
            "Process Analyst",
            "Operations Analyst",
            "Business Systems Analyst",
        ],
        "skills": [
            "Requirements Elicitation",
            "Process Mapping",
            "Gap Analysis",
            "Stakeholder Workshops",
            "KPI Framework Design",
            "Root Cause Analysis",
            "Business Case Development",
            "User Acceptance Planning",
            "Workflow Optimization",
            "Data-Driven Decision Support",
        ],
        "tools": [
            "SQL",
            "Excel",
            "Power BI",
            "Tableau",
            "Jira",
            "Confluence",
            "Visio",
            "Lucidchart",
            "Miro",
            "SAP",
        ],
        "projects": [
            "Mapped end-to-end operational workflows and reduced approval bottlenecks.",
            "Created KPI reporting framework used for monthly business reviews.",
            "Led requirements gathering and user story documentation for system modernization.",
            "Built process dashboards improving visibility into SLA and throughput metrics.",
            "Facilitated cross-team workshops to prioritize process improvement initiatives.",
            "Supported UAT execution and defect triage during platform rollout.",
        ],
        "experience_templates": [
            "Reduced process turnaround time by {metric}% through workflow redesign and handoff clarity.",
            "Captured and prioritized requirements for {count} enterprise initiatives.",
            "Improved reporting accuracy by {metric}% by standardizing KPI definitions.",
            "Facilitated stakeholder alignment workshops that accelerated project approvals.",
            "Improved UAT pass rates by {metric}% with structured scenario planning.",
        ],
        "courses": [
            "Business Analysis Foundations",
            "Process Improvement Methods",
            "Requirements Engineering",
            "Data Analytics for Business",
            "Agile Analysis Practices",
            "Decision Modeling",
        ],
    },
}

EXTRA_DOMAIN_PROFILES: dict[str, dict[str, list[str]]] = {
    "Accounting": {
        "roles": ["Accountant", "Senior Accountant", "Tax Analyst", "Audit Associate", "Payroll Specialist"],
        "skills": ["General Ledger Reconciliation", "Tax Compliance", "Accounts Payable", "Accounts Receivable", "Month-End Close", "Financial Reporting"],
        "tools": ["Tally", "QuickBooks", "SAP FICO", "Oracle Financials", "Excel", "Power BI"],
        "projects": [
            "Automated reconciliation workflow and reduced monthly close effort significantly.",
            "Improved tax filing accuracy through standardized compliance checklists.",
            "Built audit-ready documentation repository for quarterly reporting.",
            "Developed AP aging dashboard for cash-flow planning visibility.",
        ],
        "experience_templates": [
            "Reduced close cycle by {metric}% by standardizing journal and reconciliation workflows.",
            "Improved reporting accuracy by {metric}% through tighter ledger controls.",
            "Handled accounting operations for {count} entities with compliance discipline.",
        ],
        "courses": ["Advanced Accounting", "Taxation Fundamentals", "Audit and Assurance", "Corporate Reporting"],
    },
    "Healthcare": {
        "roles": ["Healthcare Analyst", "Clinical Data Coordinator", "Medical Operations Associate", "Hospital Administrator", "Patient Care Coordinator"],
        "skills": ["Clinical Documentation", "Patient Workflow Management", "Healthcare Compliance", "Care Coordination", "Medical Data Accuracy", "Operational Scheduling"],
        "tools": ["Epic", "Cerner", "Excel", "Power BI", "SQL", "Tableau"],
        "projects": [
            "Improved appointment throughput by redesigning patient intake workflows.",
            "Built quality-of-care KPI dashboards for clinical leadership reviews.",
            "Standardized records auditing to improve documentation quality.",
            "Reduced claim processing delays through process mapping and escalation tracking.",
        ],
        "experience_templates": [
            "Improved patient service turnaround by {metric}% through process redesign.",
            "Coordinated care operations across {count} departments with SLA tracking.",
            "Reduced documentation errors by {metric}% with compliance checkpoints.",
        ],
        "courses": ["Healthcare Management", "Medical Ethics", "Healthcare Analytics", "Clinical Operations"],
    },
    "Education": {
        "roles": ["Teacher", "Academic Coordinator", "Instructional Designer", "Education Program Manager", "Curriculum Specialist"],
        "skills": ["Curriculum Planning", "Classroom Delivery", "Assessment Design", "Student Performance Tracking", "Learning Program Design", "Academic Communication"],
        "tools": ["Google Classroom", "Moodle", "MS Teams", "Excel", "Canva", "PowerPoint"],
        "projects": [
            "Designed competency-based assessments and improved student outcome visibility.",
            "Created blended-learning modules for remote and in-person cohorts.",
            "Developed remedial learning framework for low-performing groups.",
            "Built academic performance tracker for semester interventions.",
        ],
        "experience_templates": [
            "Improved student engagement by {metric}% via interactive lesson strategies.",
            "Managed academic operations for {count} course cohorts.",
            "Raised assessment completion rates by {metric}% through better planning.",
        ],
        "courses": ["Curriculum Development", "Educational Psychology", "Assessment Methods", "Instructional Technology"],
    },
    "Legal": {
        "roles": ["Legal Associate", "Paralegal", "Compliance Counsel", "Contract Specialist", "Legal Analyst"],
        "skills": ["Contract Review", "Legal Research", "Regulatory Compliance", "Policy Drafting", "Case Documentation", "Risk Interpretation"],
        "tools": ["LexisNexis", "Westlaw", "DocuSign", "MS Word", "Excel", "SharePoint"],
        "projects": [
            "Created contract clause playbook to reduce review turnaround time.",
            "Built compliance tracker for policy obligations across business units.",
            "Standardized legal intake process for faster triage.",
            "Improved audit readiness through centralized legal documentation.",
        ],
        "experience_templates": [
            "Reduced contract review cycle by {metric}% using structured templates.",
            "Supported regulatory assessments across {count} internal processes.",
            "Improved compliance exception handling with formal escalation flow.",
        ],
        "courses": ["Contract Law", "Corporate Law Basics", "Compliance Management", "Legal Drafting"],
    },
    "Operations": {
        "roles": ["Operations Manager", "Operations Analyst", "Process Excellence Associate", "Business Operations Specialist", "Program Operations Lead"],
        "skills": ["Process Optimization", "SLA Management", "Workflow Governance", "Cross-Team Coordination", "Operational Reporting", "Continuous Improvement"],
        "tools": ["Excel", "Power BI", "Tableau", "Jira", "Asana", "Notion"],
        "projects": [
            "Redesigned fulfillment workflow to reduce cycle time and bottlenecks.",
            "Created weekly operations scorecards for leadership decisions.",
            "Introduced SOP repository and governance checks across teams.",
            "Implemented root-cause review process for recurring service incidents.",
        ],
        "experience_templates": [
            "Reduced operational turnaround by {metric}% through process redesign.",
            "Managed delivery SLAs across {count} business streams.",
            "Improved issue resolution consistency with standardized playbooks.",
        ],
        "courses": ["Operations Management", "Lean Basics", "Process Improvement", "Service Delivery Excellence"],
    },
    "Supply Chain": {
        "roles": ["Supply Chain Analyst", "Logistics Coordinator", "Procurement Specialist", "Inventory Planner", "Demand Planner"],
        "skills": ["Demand Planning", "Inventory Optimization", "Procurement Coordination", "Vendor Management", "Logistics Tracking", "Order Fulfillment"],
        "tools": ["SAP", "Oracle SCM", "Excel", "Power BI", "WMS", "TMS"],
        "projects": [
            "Improved forecast-to-stock alignment for critical SKUs.",
            "Built vendor performance dashboards for procurement governance.",
            "Reduced stockout incidents by introducing replenishment alerts.",
            "Standardized shipment tracking and exception management workflows.",
        ],
        "experience_templates": [
            "Reduced inventory variance by {metric}% via tighter planning controls.",
            "Coordinated procurement cycles for {count} supplier categories.",
            "Improved on-time delivery rates by {metric}% through logistics optimization.",
        ],
        "courses": ["Supply Chain Fundamentals", "Inventory Management", "Procurement Strategy", "Logistics Planning"],
    },
    "Customer Support": {
        "roles": ["Customer Support Specialist", "Support Engineer", "Customer Success Associate", "Helpdesk Analyst", "Technical Support Executive"],
        "skills": ["Ticket Triage", "Issue Resolution", "Customer Communication", "SLA Adherence", "Knowledge Base Documentation", "Escalation Handling"],
        "tools": ["Zendesk", "Freshdesk", "Intercom", "Jira Service Management", "Confluence", "Salesforce Service Cloud"],
        "projects": [
            "Created troubleshooting playbooks that improved first-response quality.",
            "Built escalation routing rules to reduce unresolved backlog.",
            "Launched self-help article library for common customer issues.",
            "Implemented CSAT monitoring dashboard for team coaching.",
        ],
        "experience_templates": [
            "Improved first-contact resolution by {metric}% through better triage workflows.",
            "Handled support operations for {count} product lines with SLA compliance.",
            "Reduced repeat tickets by {metric}% through root-cause documentation.",
        ],
        "courses": ["Customer Service Excellence", "Technical Support Foundations", "Incident Communication", "Service Operations"],
    },
    "Network Engineering": {
        "roles": ["Network Engineer", "NOC Engineer", "Infrastructure Engineer", "Network Administrator", "Systems Network Analyst"],
        "skills": ["Routing and Switching", "Network Monitoring", "Firewall Configuration", "Troubleshooting Connectivity", "LAN/WAN Design", "Infrastructure Reliability"],
        "tools": ["Cisco IOS", "Juniper", "Wireshark", "SolarWinds", "Nagios", "Fortinet"],
        "projects": [
            "Redesigned branch network topology for improved uptime and failover.",
            "Implemented network observability dashboards for proactive alerting.",
            "Reduced incident recurrence via standard change-control templates.",
            "Optimized firewall and routing policies for secure connectivity.",
        ],
        "experience_templates": [
            "Improved network uptime by {metric}% with proactive monitoring and maintenance.",
            "Managed network operations across {count} sites and service lines.",
            "Reduced incident resolution time by {metric}% through runbook standardization.",
        ],
        "courses": ["Computer Networks", "Network Security", "Routing Protocols", "Infrastructure Operations"],
    },
    "Cloud Engineering": {
        "roles": ["Cloud Engineer", "Cloud Architect", "Platform Engineer", "Site Reliability Engineer", "Infrastructure Automation Engineer"],
        "skills": ["Infrastructure as Code", "Cloud Cost Optimization", "Scalable Architecture", "CI/CD Automation", "Monitoring and Alerting", "Reliability Engineering"],
        "tools": ["AWS", "Azure", "GCP", "Terraform", "Docker", "Kubernetes"],
        "projects": [
            "Automated cloud provisioning with IaC templates across environments.",
            "Built cost observability framework for monthly spend governance.",
            "Implemented zero-downtime deployment patterns for platform services.",
            "Created reliability dashboards with SLO/SLA tracking.",
        ],
        "experience_templates": [
            "Reduced infrastructure cost by {metric}% through rightsizing and governance controls.",
            "Automated deployment workflows across {count} production services.",
            "Improved service reliability by {metric}% via resilient architecture practices.",
        ],
        "courses": ["Cloud Architecture", "Infrastructure Automation", "Site Reliability Engineering", "DevOps and CI/CD"],
    },
    "QA Testing": {
        "roles": ["QA Engineer", "Test Automation Engineer", "Quality Analyst", "SDET", "Manual Test Engineer"],
        "skills": ["Test Planning", "Regression Testing", "Automation Scripting", "Defect Tracking", "API Testing", "Quality Assurance Reporting"],
        "tools": ["Selenium", "Postman", "JMeter", "Cypress", "TestRail", "Jira"],
        "projects": [
            "Built automated regression suite for critical user journeys.",
            "Created API testing framework integrated with CI pipeline.",
            "Reduced escaped defects through risk-based test prioritization.",
            "Standardized defect lifecycle workflow across release trains.",
        ],
        "experience_templates": [
            "Improved test coverage by {metric}% through automation rollout.",
            "Reduced production defects by {metric}% via earlier quality gates.",
            "Managed QA validation cycles for {count} major releases.",
        ],
        "courses": ["Software Testing Fundamentals", "Automation Testing", "API Testing", "Quality Engineering Practices"],
    },
    "Content Writing": {
        "roles": ["Content Writer", "Technical Writer", "Content Strategist", "SEO Content Specialist", "Editorial Associate"],
        "skills": ["Long-form Writing", "Technical Documentation", "SEO Writing", "Editorial Planning", "Content Research", "Audience-Oriented Communication"],
        "tools": ["Google Docs", "Notion", "WordPress", "Grammarly", "Semrush", "Ahrefs"],
        "projects": [
            "Published high-performing content clusters aligned to search intent.",
            "Built style and tone guidelines for distributed contributors.",
            "Created product documentation that reduced support inquiries.",
            "Designed editorial calendar and content QA workflow.",
        ],
        "experience_templates": [
            "Improved content engagement by {metric}% through audience-focused writing.",
            "Produced documentation assets for {count} product capabilities.",
            "Increased search visibility by {metric}% with SEO-led content planning.",
        ],
        "courses": ["Technical Writing", "Content Strategy", "SEO Fundamentals", "Editorial Workflow Design"],
    },
    "Graphic Design": {
        "roles": ["Graphic Designer", "Visual Designer", "Brand Designer", "Marketing Designer", "Creative Designer"],
        "skills": ["Visual Composition", "Brand Identity", "Layout Design", "Typography", "Creative Direction", "Asset Production"],
        "tools": ["Adobe Photoshop", "Adobe Illustrator", "Adobe InDesign", "Figma", "Canva", "After Effects"],
        "projects": [
            "Developed cohesive brand assets for multi-channel campaigns.",
            "Redesigned marketing collaterals to improve visual consistency.",
            "Built reusable design templates for faster campaign delivery.",
            "Created motion design assets for product storytelling.",
        ],
        "experience_templates": [
            "Improved creative turnaround speed by {metric}% through reusable design systems.",
            "Delivered visual assets for {count} campaign launches.",
            "Raised brand consistency scores by {metric}% through guideline adoption.",
        ],
        "courses": ["Visual Design Principles", "Brand Design", "Typography", "Digital Illustration"],
    },
    "Hospitality": {
        "roles": ["Hotel Operations Executive", "Guest Relations Manager", "Front Office Associate", "Hospitality Supervisor", "Food and Beverage Coordinator"],
        "skills": ["Guest Service Management", "Front Desk Operations", "Reservation Handling", "Service Quality Assurance", "Complaint Resolution", "Team Coordination"],
        "tools": ["Opera PMS", "POS Systems", "Excel", "Google Workspace", "CRM", "Inventory Tools"],
        "projects": [
            "Improved guest satisfaction through streamlined check-in workflows.",
            "Built service quality checklist system for daily operations.",
            "Reduced reservation errors through standard operating templates.",
            "Optimized staffing schedules for peak occupancy periods.",
        ],
        "experience_templates": [
            "Improved guest satisfaction ratings by {metric}% via service process improvements.",
            "Managed hospitality operations for {count} shift teams.",
            "Reduced service escalations by {metric}% with proactive issue handling.",
        ],
        "courses": ["Hospitality Management", "Guest Experience Design", "Front Office Operations", "Food Service Operations"],
    },
    "Construction": {
        "roles": ["Construction Engineer", "Site Supervisor", "Project Site Coordinator", "Civil Site Engineer", "Construction Planner"],
        "skills": ["Site Execution", "Project Scheduling", "Safety Compliance", "Material Planning", "Contractor Coordination", "Progress Tracking"],
        "tools": ["AutoCAD", "MS Project", "Primavera", "Excel", "Revit", "BIM Tools"],
        "projects": [
            "Implemented site tracking dashboard for progress and risk monitoring.",
            "Improved construction schedule adherence through milestone reviews.",
            "Enhanced safety compliance through toolbox-talk governance routines.",
            "Reduced rework through better drawing/version control.",
        ],
        "experience_templates": [
            "Improved on-site schedule adherence by {metric}% via tighter planning controls.",
            "Coordinated execution across {count} contractor teams.",
            "Reduced safety incidents by {metric}% through proactive compliance checks.",
        ],
        "courses": ["Construction Project Management", "Site Safety", "Quantity Survey Basics", "Planning and Scheduling"],
    },
}

DOMAINS.update(EXTRA_DOMAIN_PROFILES)

CROSS_DOMAIN_BRIDGE_SKILLS = [
    "SQL",
    "Excel",
    "Power BI",
    "Tableau",
    "Python",
    "Jira",
    "A/B Testing",
    "Stakeholder Communication",
    "Presentation Skills",
    "Documentation",
]

PROGRAMMING_LANGUAGES_POOL = [
    "Python",
    "Java",
    "C",
    "C++",
    "C#",
    "JavaScript",
    "TypeScript",
    "Go",
    "Rust",
    "Kotlin",
    "Swift",
    "PHP",
    "Ruby",
    "R",
    "Scala",
    "MATLAB",
    "SQL",
    "Bash",
]

INSTITUTES = [
    "State University",
    "National Institute of Technology",
    "Global Business School",
    "City College",
    "Institute of Applied Sciences",
    "Tech Valley University",
    "Metropolitan University",
    "International College",
    "Center for Professional Studies",
]

DEGREES = [
    "B.Tech in Computer Science",
    "B.Sc in Statistics",
    "BBA in Marketing",
    "BA in Psychology",
    "MBA",
    "M.Sc in Data Science",
    "PG Diploma in Human Resource Management",
    "Bachelor of Commerce",
    "B.Des in Interaction Design",
    "B.Sc in Cybersecurity",
    "MBA in Finance",
]

CERTIFICATIONS = [
    "AWS Certified Cloud Practitioner",
    "Google Data Analytics Professional Certificate",
    "CompTIA Security+",
    "Certified Scrum Product Owner",
    "HubSpot Content Marketing",
    "Microsoft Power BI Data Analyst",
    "Google UX Design Certificate",
    "Certified Business Analysis Professional",
]

SUMMARY_TEMPLATES = [
    "Results-driven {role} with {years}+ years of experience delivering measurable outcomes across cross-functional teams.",
    "Detail-oriented professional in {domain}, known for balancing execution quality with business impact.",
    "Adaptive {role} with strong ownership mindset and practical experience shipping high-impact initiatives.",
    "Outcome-focused {role} skilled at collaboration, prioritization, and translating insights into action.",
]


@dataclass
class ResumeSample:
    resume_text: str
    domain: str
    domain_label: str
    multi_domain_flag: int
    secondary_domain: str
    skills: list[str]
    projects: list[str]
    courses: list[str]
    past_experience: list[str]
    education: list[str]
    languages: list[str]
    num_skills: int
    num_projects: int
    num_experiences: int
    num_courses: int
    num_languages: int
    word_count: int
    avg_sentence_length: float
    section_count: int
    skill_density: float
    metric_count: int
    metric_ratio: float
    avg_bullet_length: float
    has_summary: int
    has_projects: int
    has_experience: int
    structure_score: int
    content_score: int
    final_score: int
    classification_label: str
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]

    def to_row(self) -> dict[str, Any]:
        return {
            "resume_text": self.resume_text,
            "domain": self.domain,
            "domain_label": self.domain_label,
            "multi_domain_flag": self.multi_domain_flag,
            "secondary_domain": self.secondary_domain,
            "skills": self.skills,
            "projects": self.projects,
            "courses": self.courses,
            "past_experience": self.past_experience,
            "education": self.education,
            "languages": self.languages,
            "num_skills": self.num_skills,
            "num_projects": self.num_projects,
            "num_experiences": self.num_experiences,
            "num_courses": self.num_courses,
            "num_languages": self.num_languages,
            "word_count": self.word_count,
            "avg_sentence_length": self.avg_sentence_length,
            "section_count": self.section_count,
            "skill_density": self.skill_density,
            "metric_count": self.metric_count,
            "metric_ratio": self.metric_ratio,
            "avg_bullet_length": self.avg_bullet_length,
            "has_summary": self.has_summary,
            "has_projects": self.has_projects,
            "has_experience": self.has_experience,
            "structure_score": self.structure_score,
            "content_score": self.content_score,
            "final_score": self.final_score,
            "classification_label": self.classification_label,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
        }


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def pick_count(rng: random.Random, quality: str, low: int, medium: int, high: int) -> int:
    if quality == "good":
        return rng.randint(medium, high)
    if quality == "average":
        return rng.randint(max(1, low), medium)
    return rng.randint(0, max(1, low))


def maybe_typo(text: str, rng: random.Random, intensity: float) -> str:
    if intensity <= 0:
        return text

    words = text.split()
    if not words:
        return text

    typo_ops = ["drop", "swap", "duplicate"]
    edits = max(1, int(len(words) * intensity))

    for _ in range(edits):
        idx = rng.randrange(len(words))
        word = words[idx]
        if len(word) < 4 or rng.random() > 0.45:
            continue

        op = rng.choice(typo_ops)
        chars = list(word)

        if op == "drop" and len(chars) > 4:
            del chars[rng.randrange(1, len(chars) - 1)]
        elif op == "swap" and len(chars) > 4:
            j = rng.randrange(1, len(chars) - 2)
            chars[j], chars[j + 1] = chars[j + 1], chars[j]
        elif op == "duplicate" and len(chars) > 3:
            j = rng.randrange(1, len(chars) - 1)
            chars.insert(j, chars[j])

        words[idx] = "".join(chars)

    return " ".join(words)


def sentence_stats(text: str) -> tuple[int, float]:
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return word_count, 0.0

    sent_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    avg_len = sum(sent_lengths) / len(sent_lengths)
    return word_count, round(avg_len, 2)


def metric_stats(text: str) -> tuple[int, float]:
    metrics = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
    metric_count = len(metrics)
    word_count, _ = sentence_stats(text)
    metric_ratio = metric_count / max(word_count, 1)
    return metric_count, round(metric_ratio, 4)


def avg_bullet_length(text: str) -> float:
    bullets = [
        re.sub(r"^[-*\u2022\s]+", "", line).strip()
        for line in text.splitlines()
        if re.match(r"^\s*[-*\u2022]", line)
    ]
    if not bullets:
        return 0.0
    lengths = [len(re.findall(r"\b\w+\b", bullet)) for bullet in bullets if bullet]
    return round(sum(lengths) / len(lengths), 2) if lengths else 0.0


def choose_sections(rng: random.Random, quality: str) -> dict[str, bool]:
    sections = {
        "Summary": rng.random() < 0.78,
        "Skills": True,
        "Projects": True,
        "Experience": True,
        "Education": True,
        "Courses": True,
        "Languages": True,
    }

    if quality == "average":
        for key in ["Summary", "Courses", "Languages", "Projects"]:
            if rng.random() < 0.24:
                sections[key] = False

    if quality == "bad":
        for key in ["Summary", "Projects", "Courses", "Languages", "Experience", "Education"]:
            if rng.random() < 0.46:
                sections[key] = False
        if not any(sections.values()):
            sections["Skills"] = True

    return sections


def fill_experience_template(template: str, rng: random.Random) -> str:
    return template.format(
        metric=rng.randint(10, 48),
        count=rng.randint(2, 12),
        from_time=rng.randint(30, 110),
        to_time=rng.randint(12, 60),
    )


def build_experience(
    role: str,
    primary_domain: str,
    primary_profile: dict[str, list[str]],
    rng: random.Random,
    quality: str,
    secondary_role: str | None,
    secondary_profile: dict[str, list[str]] | None,
) -> list[str]:
    exp_count = pick_count(rng, quality, low=1, medium=2, high=4)
    entries: list[str] = []

    for i in range(exp_count):
        use_secondary = secondary_profile is not None and i == exp_count - 1 and rng.random() < 0.42
        if use_secondary and secondary_profile and secondary_role:
            template = rng.choice(secondary_profile["experience_templates"])
            action = fill_experience_template(template, rng)
            role_name = secondary_role
            domain_tag = "cross-domain"
        else:
            template = rng.choice(primary_profile["experience_templates"])
            action = fill_experience_template(template, rng)
            role_name = role
            domain_tag = primary_domain

        company = f"{rng.choice(['Apex', 'Nexa', 'Vertex', 'BlueWave', 'Bright', 'Orbit'])} {rng.choice(['Solutions', 'Systems', 'Labs', 'Technologies', 'Group'])}"
        start_year = 2015 + rng.randint(0, 8)
        end_year = start_year + rng.randint(1, 3)
        entries.append(
            f"{role_name} at {company} ({start_year}-{end_year}) [{domain_tag}]: {action}"
        )

    return entries


def render_resume_text(
    name: str,
    primary_domain: str,
    role: str,
    sections: dict[str, bool],
    summary: str,
    skills: list[str],
    projects: list[str],
    experiences: list[str],
    education: list[str],
    courses: list[str],
    languages: list[str],
    rng: random.Random,
    quality: str,
    secondary_domain: str | None,
) -> str:
    domain_line = primary_domain if secondary_domain is None else f"{primary_domain} + {secondary_domain}"
    clean_header = f"{name}\nTarget Role: {role} | Domain: {domain_line}\n"
    messy_header = (
        f"{name.upper()} :: {role} -- {domain_line}\n"
        f"mail maybe: candidate{rng.randint(100,999)}@mail.com | phone: +1-{rng.randint(200,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}\n"
    )

    if quality == "bad" and rng.random() < 0.65:
        text = messy_header
    else:
        text = clean_header + (
            f"Email: {name.lower().replace(' ','.')}@example.com\n"
            f"Phone: +1-{rng.randint(200,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}\n"
        )

    section_styles = [
        lambda s: s.upper(),
        lambda s: s,
        lambda s: f"[{s}]",
        lambda s: f"## {s}",
    ]

    if quality == "good":
        style_fn = section_styles[rng.randint(0, 1)]
        bullet = "- "
        gap = "\n\n"
    elif quality == "average":
        style_fn = rng.choice(section_styles[:3])
        bullet = rng.choice(["- ", "* "])
        gap = "\n"
    else:
        style_fn = rng.choice(section_styles)
        bullet = rng.choice(["", "- ", "* ", "> "])
        gap = "\n"

    def add_section(title: str, lines: list[str]) -> str:
        body = "\n".join(f"{bullet}{line}" if bullet else line for line in lines)
        return f"{style_fn(title)}\n{body}"

    blocks: list[str] = []
    if sections.get("Summary") and summary:
        blocks.append(add_section("Summary", [summary]))
    if sections.get("Skills") and skills:
        blocks.append(add_section("Skills", skills))
    if sections.get("Projects") and projects:
        blocks.append(add_section("Projects", projects))
    if sections.get("Experience") and experiences:
        blocks.append(add_section("Experience", experiences))
    if sections.get("Education") and education:
        blocks.append(add_section("Education", education))
    if sections.get("Courses") and courses:
        blocks.append(add_section("Courses", courses))
    if sections.get("Languages") and languages:
        blocks.append(add_section("Languages", languages))

    if quality == "bad" and rng.random() < 0.55:
        rng.shuffle(blocks)

    text += gap.join(blocks)

    typo_intensity = {"good": 0.0, "average": 0.012, "bad": 0.055}[quality]
    text = maybe_typo(text, rng, typo_intensity)

    if quality == "bad" and rng.random() < 0.44:
        text = text.replace("\n", " ")

    return text.strip()


def score_structure(
    sections: dict[str, bool],
    quality: str,
    word_count: int,
    avg_sentence_length: float,
    text: str,
) -> int:
    required = ["Skills", "Projects", "Experience", "Education", "Courses", "Languages"]
    present_required = sum(1 for sec in required if sections.get(sec, False))
    section_presence_score = (present_required / len(required)) * 45

    headings = sum(
        1
        for marker in ["SUMMARY", "Skills", "Projects", "Experience", "Education", "Courses", "Languages", "##"]
        if marker in text
    )
    formatting_score = min(25.0, headings * 3.8)

    if avg_sentence_length <= 0:
        readability_score = 0.0
    else:
        distance = abs(avg_sentence_length - 17)
        readability_score = max(0.0, 20.0 - distance * 1.4)

    organization_score = 10.0
    if quality == "average":
        organization_score -= 2.5
    if quality == "bad":
        organization_score -= 6.0
    if word_count < 110:
        organization_score -= 2.0

    raw = section_presence_score + formatting_score + readability_score + organization_score
    return int(max(0, min(100, round(raw))))


def score_content(
    num_skills: int,
    num_projects: int,
    num_experiences: int,
    num_courses: int,
    num_languages: int,
    text: str,
) -> int:
    skill_component = min(25.0, num_skills * 2.4)
    project_component = min(25.0, num_projects * 7.0)
    experience_component = min(30.0, num_experiences * 9.0)
    course_lang_component = min(12.0, num_courses * 2.4 + num_languages * 1.8)

    richness_bonus = 0.0
    if len(text) > 700:
        richness_bonus += 4.0
    if re.search(r"\b(improved|increased|reduced|achieved|built|launched|optimized|implemented)\b", text, flags=re.IGNORECASE):
        richness_bonus += 4.0

    raw = skill_component + project_component + experience_component + course_lang_component + richness_bonus
    return int(max(0, min(100, round(raw))))


def classify(final_score: int) -> str:
    if final_score >= 75:
        return "good"
    if final_score >= 50:
        return "average"
    return "bad"


def build_feedback(
    structure_score: int,
    content_score: int,
    num_skills: int,
    num_projects: int,
    num_experiences: int,
    num_courses: int,
    num_languages: int,
    section_count: int,
) -> tuple[list[str], list[str], list[str]]:
    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []

    if structure_score >= 75:
        strengths.append("Resume structure is clear with identifiable sections.")
    elif structure_score >= 55:
        strengths.append("Resume has partial structure and can be scanned reasonably well.")
    else:
        weaknesses.append("Resume structure is weak and difficult to navigate quickly.")

    if content_score >= 75:
        strengths.append("Content depth is strong with good evidence of capabilities.")
    elif content_score >= 55:
        strengths.append("Content quality is moderate and can be improved with stronger details.")
    else:
        weaknesses.append("Content appears thin and lacks enough supporting details.")

    if num_projects == 0:
        weaknesses.append("No projects listed.")
        suggestions.append("Add 2-3 projects with measurable outcomes and technologies used.")
    elif num_projects == 1:
        suggestions.append("Expand project section with additional practical work.")

    if num_skills < 5:
        weaknesses.append("Skill coverage is limited.")
        suggestions.append("Improve skillset breadth by adding relevant domain and tool-specific skills.")

    if num_experiences == 0:
        weaknesses.append("No prior experience entries found.")
        suggestions.append("Add internships, freelancing, or volunteer experience to show practical exposure.")

    if section_count < 4:
        weaknesses.append("Several key resume sections are missing.")
        suggestions.append("Use clear section headings for Skills, Experience, Projects, Education, Courses, and Languages.")

    if structure_score < 60:
        suggestions.append("Improve formatting with consistent headings, bullet points, and spacing.")

    if num_courses == 0:
        suggestions.append("Include relevant courses or certifications to strengthen profile credibility.")

    if num_languages == 0:
        suggestions.append("Add programming languages relevant to your target role.")

    if not strengths:
        strengths.append("Resume includes at least some professional details and role context.")

    if not weaknesses:
        weaknesses.append("No major weaknesses detected.")

    if not suggestions:
        suggestions.append("Tailor achievements to target roles and quantify impact wherever possible.")

    strengths = list(dict.fromkeys(strengths))[:4]
    weaknesses = list(dict.fromkeys(weaknesses))[:5]
    suggestions = list(dict.fromkeys(suggestions))[:5]
    return strengths, weaknesses, suggestions


def choose_programming_languages(
    rng: random.Random,
    skills: list[str],
    primary_profile: dict[str, list[str]],
    secondary_profile: dict[str, list[str]] | None,
    target_count: int,
) -> list[str]:
    direct_candidates = [
        item
        for item in dedupe_keep_order(skills + primary_profile["tools"] + (secondary_profile["tools"] if secondary_profile else []))
        if item in PROGRAMMING_LANGUAGES_POOL
    ]

    if len(direct_candidates) < target_count:
        remaining = [x for x in PROGRAMMING_LANGUAGES_POOL if x not in direct_candidates]
        rng.shuffle(remaining)
        direct_candidates.extend(remaining[: max(0, target_count - len(direct_candidates))])

    return direct_candidates[:target_count]


def generate_single_resume(
    rng: random.Random,
    quality: str,
    primary_domain: str,
    multi_domain_probability: float,
) -> ResumeSample:
    primary_profile = DOMAINS[primary_domain]
    role = rng.choice(primary_profile["roles"])

    secondary_domain: str | None = None
    secondary_profile: dict[str, list[str]] | None = None
    secondary_role: str | None = None

    if rng.random() < multi_domain_probability:
        secondary_domain = rng.choice([d for d in DOMAINS.keys() if d != primary_domain])
        secondary_profile = DOMAINS[secondary_domain]
        secondary_role = rng.choice(secondary_profile["roles"])

    name = (
        f"{rng.choice(['Alex', 'Jordan', 'Taylor', 'Riley', 'Morgan', 'Avery', 'Casey', 'Sam', 'Jamie', 'Drew'])} "
        f"{rng.choice(['Smith', 'Patel', 'Kim', 'Garcia', 'Khan', 'Johnson', 'Nguyen', 'Brown', 'Singh', 'Lee'])}"
    )
    sections = choose_sections(rng, quality)

    target_skill_count = pick_count(rng, quality, low=5, medium=9, high=15)
    target_tool_count = pick_count(rng, quality, low=2, medium=4, high=7)
    target_project_count = pick_count(rng, quality, low=1, medium=2, high=5)
    target_course_count = pick_count(rng, quality, low=1, medium=2, high=4)
    target_lang_count = pick_count(rng, quality, low=1, medium=2, high=4)

    primary_skills = rng.sample(primary_profile["skills"], k=min(target_skill_count, len(primary_profile["skills"])))
    primary_tools = rng.sample(primary_profile["tools"], k=min(target_tool_count, len(primary_profile["tools"])))

    blended_skills = primary_skills + primary_tools
    blended_projects = rng.sample(primary_profile["projects"], k=min(target_project_count, len(primary_profile["projects"])))
    blended_courses = rng.sample(primary_profile["courses"], k=min(target_course_count, len(primary_profile["courses"])))

    if secondary_profile is not None:
        sec_skill_take = min(rng.randint(2, 4), len(secondary_profile["skills"]))
        sec_tool_take = min(rng.randint(1, 3), len(secondary_profile["tools"]))
        sec_project_take = min(rng.randint(1, 2), len(secondary_profile["projects"]))
        sec_course_take = min(rng.randint(0, 2), len(secondary_profile["courses"]))

        blended_skills.extend(rng.sample(secondary_profile["skills"], k=sec_skill_take))
        blended_skills.extend(rng.sample(secondary_profile["tools"], k=sec_tool_take))
        blended_projects.extend(rng.sample(secondary_profile["projects"], k=sec_project_take))
        if sec_course_take > 0:
            blended_courses.extend(rng.sample(secondary_profile["courses"], k=sec_course_take))

        overlap_take = rng.randint(1, 3)
        blended_skills.extend(rng.sample(CROSS_DOMAIN_BRIDGE_SKILLS, k=overlap_take))

    skills = dedupe_keep_order(blended_skills)[: max(0, target_skill_count + target_tool_count)]
    projects = dedupe_keep_order(blended_projects)[: max(0, target_project_count + 1)]
    courses = dedupe_keep_order(blended_courses)[: max(0, target_course_count + 1)]

    experiences = build_experience(
        role=role,
        primary_domain=primary_domain,
        primary_profile=primary_profile,
        rng=rng,
        quality=quality,
        secondary_role=secondary_role,
        secondary_profile=secondary_profile,
    )

    education: list[str] = []
    edu_count = pick_count(rng, quality, low=1, medium=1, high=2)
    for _ in range(edu_count):
        degree = rng.choice(DEGREES)
        school = rng.choice(INSTITUTES)
        grad_year = rng.randint(2012, 2025)
        education.append(f"{degree} - {school} ({grad_year})")
    if quality != "bad" and rng.random() < 0.35:
        education.append(rng.choice(CERTIFICATIONS))

    languages = choose_programming_languages(
        rng=rng,
        skills=skills,
        primary_profile=primary_profile,
        secondary_profile=secondary_profile,
        target_count=target_lang_count,
    )

    summary = rng.choice(SUMMARY_TEMPLATES).format(role=role, domain=primary_domain, years=rng.randint(1, 10))
    if secondary_domain is not None and rng.random() < 0.75:
        summary += f" Cross-domain exposure in {secondary_domain} environments."

    if not sections.get("Skills"):
        skills = []
    if not sections.get("Projects"):
        projects = []
    if not sections.get("Experience"):
        experiences = []
    if not sections.get("Education"):
        education = []
    if not sections.get("Courses"):
        courses = []
    if not sections.get("Languages"):
        languages = []
    if not sections.get("Summary"):
        summary = ""

    text = render_resume_text(
        name=name,
        primary_domain=primary_domain,
        role=role,
        sections=sections,
        summary=summary,
        skills=skills,
        projects=projects,
        experiences=experiences,
        education=education,
        courses=courses,
        languages=languages,
        rng=rng,
        quality=quality,
        secondary_domain=secondary_domain,
    )

    word_count, avg_sentence_length = sentence_stats(text)
    metric_count, metric_ratio = metric_stats(text)
    bullet_length = avg_bullet_length(text)
    section_count = sum(1 for items in [skills, projects, experiences, education, courses, languages] if items) + (1 if summary else 0)
    skill_density = len(skills) / max(word_count, 1)

    structure_score = score_structure(sections, quality, word_count, avg_sentence_length, text)
    content_score = score_content(
        num_skills=len(skills),
        num_projects=len(projects),
        num_experiences=len(experiences),
        num_courses=len(courses),
        num_languages=len(languages),
        text=text,
    )
    final_score = int(round(0.4 * structure_score + 0.6 * content_score))
    classification_label = classify(final_score)

    strengths, weaknesses, suggestions = build_feedback(
        structure_score=structure_score,
        content_score=content_score,
        num_skills=len(skills),
        num_projects=len(projects),
        num_experiences=len(experiences),
        num_courses=len(courses),
        num_languages=len(languages),
        section_count=section_count,
    )

    return ResumeSample(
        resume_text=text,
        domain=primary_domain,
        domain_label=primary_domain,
        multi_domain_flag=1 if secondary_domain else 0,
        secondary_domain=secondary_domain or "",
        skills=skills,
        projects=projects,
        courses=courses,
        past_experience=experiences,
        education=education,
        languages=languages,
        num_skills=len(skills),
        num_projects=len(projects),
        num_experiences=len(experiences),
        num_courses=len(courses),
        num_languages=len(languages),
        word_count=word_count,
        avg_sentence_length=avg_sentence_length,
        section_count=section_count,
        skill_density=round(skill_density, 4),
        metric_count=metric_count,
        metric_ratio=metric_ratio,
        avg_bullet_length=bullet_length,
        has_summary=1 if sections.get("Summary") else 0,
        has_projects=1 if sections.get("Projects") else 0,
        has_experience=1 if sections.get("Experience") else 0,
        structure_score=structure_score,
        content_score=content_score,
        final_score=final_score,
        classification_label=classification_label,
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
    )


def generate_balanced_targets(sample_count: int) -> dict[str, int]:
    domains = list(DOMAINS.keys())
    n_domains = len(domains)
    base = sample_count // n_domains
    remainder = sample_count % n_domains
    return {domain: base + (1 if i < remainder else 0) for i, domain in enumerate(domains)}


def generate_dataset(sample_count: int, seed: int, multi_domain_ratio: float = 0.22) -> list[ResumeSample]:
    rng = random.Random(seed)
    quality_choices = ["good", "average", "bad"]
    quality_weights = [0.4, 0.35, 0.25]

    per_domain_targets = generate_balanced_targets(sample_count)
    if min(per_domain_targets.values()) < 50:
        raise ValueError("Sample size too small for current domain coverage. Increase --samples to keep domain diversity.")

    data: list[ResumeSample] = []
    for domain, target_count in per_domain_targets.items():
        for _ in range(target_count):
            quality = rng.choices(quality_choices, weights=quality_weights, k=1)[0]
            data.append(
                generate_single_resume(
                    rng=rng,
                    quality=quality,
                    primary_domain=domain,
                    multi_domain_probability=multi_domain_ratio,
                )
            )

    rng.shuffle(data)
    return data


def write_csv(samples: list[ResumeSample], path: Path) -> None:
    rows = [sample.to_row() for sample in samples]
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            for key in ["skills", "projects", "courses", "past_experience", "education", "languages", "strengths", "weaknesses", "suggestions"]:
                out[key] = json.dumps(out[key], ensure_ascii=False)
            writer.writerow(out)


def write_json(samples: list[ResumeSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [sample.to_row() for sample in samples]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def summarize_distribution(samples: list[ResumeSample]) -> dict[str, Any]:
    domain_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    multi_domain_count = 0

    for sample in samples:
        domain_counts[sample.domain] = domain_counts.get(sample.domain, 0) + 1
        quality_counts[sample.classification_label] = quality_counts.get(sample.classification_label, 0) + 1
        multi_domain_count += sample.multi_domain_flag

    return {
        "domains": dict(sorted(domain_counts.items())),
        "labels": dict(sorted(quality_counts.items())),
        "multi_domain_count": multi_domain_count,
        "multi_domain_ratio": round(multi_domain_count / max(1, len(samples)), 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic resume dataset.")
    parser.add_argument("--samples", type=int, default=1500, help="Number of resume samples (required 1500-2000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--out-dir", type=str, default="data/datasets/new", help="Output directory for generated dataset files.")
    parser.add_argument("--multi-domain-ratio", type=float, default=0.22, help="Probability that a resume includes cross-domain signals.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1500 or args.samples > 2000:
        raise ValueError("Please set --samples between 1500 and 2000.")

    out_dir = Path(args.out_dir)
    csv_path = out_dir / "synthetic_resumes.csv"
    json_path = out_dir / "synthetic_resumes.json"

    samples = generate_dataset(
        sample_count=args.samples,
        seed=args.seed,
        multi_domain_ratio=max(0.0, min(0.7, args.multi_domain_ratio)),
    )
    write_csv(samples, csv_path)
    write_json(samples, json_path)

    distribution = summarize_distribution(samples)

    print(f"Generated {len(samples)} synthetic resumes.")
    print(f"CSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    print("\nDomain distribution summary:")
    print(json.dumps(distribution, indent=2, ensure_ascii=False))

    preview_rows = [samples[i].to_row() for i in range(min(3, len(samples)))]
    print("\nPreview (first 3 samples):")
    print(json.dumps(preview_rows, indent=2, ensure_ascii=False)[:3600])


if __name__ == "__main__":
    main()
