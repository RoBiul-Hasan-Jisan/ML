// import React from "react";
import { useNavigate } from "react-router-dom";

import WhatIsML from "./MLIntroductionExtra/WhatIsML";
import MLvsTraditional from "./MLIntroductionExtra/MLvsTraditional";
import WhyML from "./MLIntroductionExtra/WhyML";
import HowMLWorks from "./MLIntroductionExtra/HowMLWorks";
import MLLifecycle from "./MLIntroductionExtra/MLLifecycle";
import TypesOfML from "./MLIntroductionExtra/TypesOfML";

export default function MLIntroduction() {
  const navigate = useNavigate();

  const goTo = (path: string): void => {
    navigate(path);
  };

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-12">
      <WhatIsML />
      <MLvsTraditional />
      <WhyML />
      <HowMLWorks />
      <MLLifecycle />
      <TypesOfML goTo={goTo} />
    </div>
  );
}
