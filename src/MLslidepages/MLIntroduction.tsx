//import React from "react";
import { useNavigate } from "react-router-dom";

import WhatIsML from "./MLIntroductionExtra/WhatIsML.tsx";
import MLvsTraditional from "./MLIntroductionExtra/MLvsTraditional.tsx";
import WhyML from "./MLIntroductionExtra/WhyML.tsx";
import HowMLWorks from "./MLIntroductionExtra/HowMLWorks.tsx";
import MLLifecycle from "./MLIntroductionExtra/MLLifecycle.tsx";
import TypesOfML from "./MLIntroductionExtra/TypesOfML.tsx";

export default function MLIntroduction() {
  const navigate = useNavigate();

  // Type the parameter as string, since navigate expects string paths
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
      {/* Pass the goTo function as a prop */}
      <TypesOfML goTo={goTo} />
    </div>
  );
}
